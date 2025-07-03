import numpy as np
import pandapower.networks as pn
from pandapower import runpp
import pandas as pd
import pyomo.environ as pyo
import copy

from optimal_power_flow import Model

class Poly_AC_OPF(Model):

    def __init__(self, network, settings=None):
        super().__init__(network, settings)

        # Pyomo Model
        self.model = None

    def build_model(self):

        # Instantiate model
        model = pyo.ConcreteModel()
        self.model = model

        # Create sets
        model.buses = pyo.Set(initialize=list(self.buses.keys()))
        model.generators = pyo.Set(initialize=list(self.generators.keys()))
        model.lines = pyo.Set(initialize=list(self.lines.keys()))
        self.slack = self.net.ext_grid.bus.values[0]

        # Create parameters
        model.P_load = pyo.Param(model.buses, initialize={i: bus.pD for i, bus in self.buses.items()}, mutable=True)
        model.Q_load = pyo.Param(model.buses, initialize={i: bus.qD for i, bus in self.buses.items()}, mutable=True)

        # Create variables
        model.V_re = pyo.Var(model.buses, within=pyo.Reals, initialize=1.0)
        model.V_im = pyo.Var(model.buses, within=pyo.Reals, initialize=0.0)

        # Generator outputs (only for generator buses)
        model.P_gen = pyo.Var(model.generators, within=pyo.Reals, initialize=0.3)
        model.Q_gen = pyo.Var(model.generators, within=pyo.Reals, initialize=0.1)

        # Create expressions

        # Net injections = generation - load

        # Write constraints

        # Power flow balance constraints
        def P_balance_rule(m, k):
            bus = self.buses[k]
            P_gen = sum(m.P_gen[g.name] for g in bus.generators)
            P_flow = m.V_re[k] * sum(self.G[k, i] * m.V_re[i] - self.B[k, i] * m.V_im[i] for i in m.buses) \
                     + m.V_im[k] * sum(self.B[k, i] * m.V_re[i] + self.G[k, i] * m.V_im[i] for i in m.buses)
            return P_gen == P_flow + m.P_load[k]

        model.P_balance = pyo.Constraint(model.buses, rule=P_balance_rule)

        def Q_balance_rule(m, k):
            bus = self.buses[k]
            Q_gen = sum(m.Q_gen[g.name] for g in bus.generators)
            Q_flow = m.V_re[k] * sum(-self.B[k, i] * m.V_re[i] - self.G[k, i] * m.V_im[i] for i in m.buses) \
                     + m.V_im[k] * sum(self.G[k, i] * m.V_re[i] - self.B[k, i] * m.V_im[i] for i in m.buses)
            return Q_gen == Q_flow + m.Q_load[k]

        model.Q_balance = pyo.Constraint(model.buses, rule=Q_balance_rule)

        # Voltage magnitude constraints
        def V_mag_rule(m, k):
            bus = self.buses[k]
            return pyo.inequality(bus.vmin ** 2, m.V_re[k] ** 2 + m.V_im[k] ** 2, bus.vmax ** 2)

        model.V_mag = pyo.Constraint(model.buses, rule=V_mag_rule)

        # Fix slack bus voltage and voltage angle
        model.V_re[self.slack].fix(1.0)
        model.V_im[self.slack].fix(0.0)

        # Generator limits
        def P_gen_limits_rule(m, g):
            gen = self.generators[g]
            return pyo.inequality(gen.pmin, m.P_gen[g], gen.pmax)

        model.P_gen_limits = pyo.Constraint(model.generators, rule=P_gen_limits_rule)

        def Q_gen_limits_rule(m, g):
            gen = self.generators[g]
            return pyo.inequality(gen.qmin, m.Q_gen[g], gen.qmax)

        model.Q_gen_limits = pyo.Constraint(model.generators, rule=Q_gen_limits_rule)

        # Set objective
        def objective_rule(m):
            return sum(
                gen.cost[0] + gen.cost[1] * (gen.baseMVA * m.P_gen[i]) + gen.cost[2] * (gen.baseMVA * m.P_gen[i]) ** 2
                for i, gen in self.generators.items())

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def initialize_model(self):
        # Run power flow
        runpp(net, numba=False)

        # Extract complex bus voltages
        vm = net.res_bus.vm_pu  # per unit voltage magnitude
        va = np.deg2rad(net.res_bus.va_degree)  # voltage angle in degrees
        v = vm * np.exp(1j * va)
        v_re, v_im = v.map(np.real), v.map(np.imag)

        # Extract complex generator power injections
        pg = net.res_gen.p_mw
        qg = net.res_gen.q_mvar

        # Initialize values
        for k in self.model.buses:
            self.model.V_re[k].value = v_re.loc[k]
            self.model.V_im[k].value = v_im.loc[k]

        # Initialize values
        for g in self.model.generators:
            self.model.P_gen[g].value = pg.loc[g]
            self.model.Q_gen[g].value = qg.loc[g]

    def solve_model(self):
        solver = pyo.SolverFactory('ipopt')
        solver.solve(self.model, tee=True)

    def print_solution(self):
        # Display results
        print("\n=== Generator Dispatch (MW) ===")
        for gen in self.generators.values():
            print(
                f"Generator {gen.name} at bus {gen.bus.name}: P = {pyo.value(self.model.P_gen[gen.name]) * gen.baseMVA:.2f} MW, Q = {pyo.value(self.model.Q_gen[gen.name]) * gen.baseMVA:.2f} MVar")

        print("\n=== Bus Voltages (p.u.) ===")
        for bus in self.buses.values():
            V_re = pyo.value(self.model.V_re[bus.name])
            V_im = pyo.value(self.model.V_im[bus.name])
            V_mag = np.sqrt(V_re ** 2 + V_im ** 2)
            V_angle = np.degrees(np.arctan2(V_im, V_re))
            print(f"Bus {bus.name}: |V| = {V_mag:.4f} p.u., angle = {V_angle:.2f}°")