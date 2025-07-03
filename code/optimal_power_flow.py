import numpy as np
import pandapower.networks as pn
from pandapower import runpp
import pandas as pd
import pyomo.environ as pyo
import copy

class Component: # Basic component in power system optimization model

    def __init__(self, **kwargs):
        self.baseMVA = 100
        pass

    def create_parameters(self):
        pass

    def update_timeseries_parameters(self, date):
        pass

    def create_variables(self):
        pass

    def create_expressions(self):
        pass

class Bus(Component):

    def __init__(self, name: str, data: pd.Series, load_data: pd.Series, shunt_data: pd.Series):
        super().__init__()

        self.name = name
        self.type = data["type"]
        self.vbase = data["vn_kv"]
        self.vmax = 1.2  # data["max_vm_pu"]
        self.vmin = 0.8  # data["min_vm_pu"]
        self.pD = load_data["p_mw"] / self.baseMVA
        self.qD = load_data["q_mvar"] / self.baseMVA
        self.qShunt = shunt_data["q_mvar"] / self.baseMVA

        self.generators = []
        self.in_lines = []
        self.out_lines = []

    @classmethod
    def from_series(cls, data: pd.Series, load_data: pd.Series, shunt_data: pd.Series):
        return cls(data.name, data, load_data, shunt_data)

class Generator(Component):

    def __init__(self, name: str, data: pd.Series, cost_data: pd.Series):
        super().__init__()

        self.name = name
        self.bus_ID = data["bus"]
        self.pmin = data["min_p_mw"] / self.baseMVA
        self.pmax = data["max_p_mw"] / self.baseMVA
        self.qmin = data["min_q_mvar"] / self.baseMVA
        self.qmax = data["max_q_mvar"] / self.baseMVA
        self.cost = cost_data[["cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]].values

        self.bus = None

    @classmethod
    def from_series(cls, data: pd.Series, cost_data: pd.Series):
        return cls(data.name, data, cost_data)

    def link_bus(self, buses: dict):
        # Link node to resource
        self.bus = buses[self.bus_ID]
        # Link resource to node
        buses[self.bus_ID].generators.append(self)

class Line(Component):

    def __init__(self, name: str, data: pd.Series):
        super().__init__()

        self.name = name
        self.from_bus_ID = data["from_bus"]
        self.to_bus_ID = data["to_bus"]
        self.r_ohm = data["length_km"] * data["r_ohm_per_km"]
        self.x_ohm = data["length_km"] * data["x_ohm_per_km"]
        self.imax = data["max_i_ka"]

        # Per unit line impedance / admittance values
        self.z, self.r, self.x, self.y, self.g, self.b = None, None, None, None, None, None

        self.from_bus = None
        self.to_bus = None

    def calculate_admittance(self):
        # Calculate base impedance (ohms)
        vbase = self.from_bus.vbase * 1e3  # convert kV to V
        sbase = self.baseMVA * 1e6  # convert MVA to VA
        zbase = vbase ** 2 / sbase

        assert abs(self.from_bus.vbase - self.to_bus.vbase) < 1e-3, "Voltage base mismatch across line ends."

        # Convert r and x to per unit
        self.r = self.r_ohm / zbase
        self.x = self.x_ohm / zbase
        self.z = complex(self.r, self.x)  # Impedance

        # Calculate y, g and b
        self.y = 1 / self.z if abs(self.z) > 0 else 0  # Admittance
        self.g, self.b = np.real(self.y), np.imag(self.y)

    @classmethod
    def from_series(cls, data: pd.Series):
        return cls(data.name, data)

    def link_buses(self, buses: dict):
        # Link buses to line
        self.from_bus = buses[self.from_bus_ID]
        self.to_bus = buses[self.to_bus_ID]
        # Link line to nodes
        buses[self.from_bus_ID].out_lines.append(self)
        buses[self.to_bus_ID].in_lines.append(self)


class Model:

    def __init__(self, network, settings=None):

        # PandaPower network and model settings
        self.net = network
        self.settings = settings
        self.baseMVA = network.sn_mva

        # Power system components
        self.buses = {}
        self.generators = {}
        self.lines = {}

    @property
    def components(self):
        return list(self.buses.values() + self.generators.values() + self.lines.values())

    def create_system(self):

        # Pre-process data
        net = copy.copy(self.net)
        # Add loads to all buses
        net.load = net.load.set_index("bus").reindex(net.bus.index)
        net.load[["p_mw", "q_mvar"]] = net.load[["p_mw", "q_mvar"]].fillna(0)
        # Drop duplicate cost data
        net.poly_cost = net.poly_cost.drop_duplicates(subset=["element"]).set_index("element")
        # Add shunts to all busees
        net.shunt = net.shunt.set_index("bus").reindex(net.bus.index)
        net.shunt["q_mvar"] = net.shunt["q_mvar"].fillna(0)

        # Create buses, generators, and line instances from network data
        for bus in net.bus.index:
            self.buses[bus] = Bus.from_series(net.bus.loc[bus], net.load.loc[bus], net.shunt.loc[bus])

        for gen in net.gen.index:
            self.generators[gen] = Generator.from_series(net.gen.loc[gen], net.poly_cost.loc[gen])
            self.generators[gen].link_bus(self.buses)  # Link generator to node

        for line in net.line.index:
            self.lines[line] = Line.from_series(net.line.loc[line])
            self.lines[line].link_buses(self.buses)  # Link line to nodes
            self.lines[line].calculate_admittance()  # Calculate admittance (g, b) using from_bus base voltage

        # Create admittance matrix
        self.Y = np.zeros((len(self.buses), len(self.buses)), dtype=complex)
        for line in self.lines.values():
            k = line.from_bus.name
            i = line.to_bus.name
            y = line.y
            self.Y[k, k] += y
            self.Y[i, i] += y
            self.Y[k, i] -= y
            self.Y[i, k] -= y
        # Add shunts
        for bus in self.buses.values():
            k = bus.name
            self.Y[k, k] -= 1j * bus.qShunt
        # Split real and imaginary components
        self.G = np.real(self.Y)
        self.B = np.imag(self.Y)

    def build_model(self):
        pass

    def initialize_model(self):
        pass
    def solve_model(self):
        pass

    def print_solution(self):
        pass