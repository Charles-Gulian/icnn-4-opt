clear settings;
settings.alpha = 1;
settings.line_prob = 'all';
settings.epL = 10000;
settings.tol_feas = 1.5 * 10^(-5);

% Define case name
case_name = 'case2383wp';
model = eval(case_name);
num_buses = size(model.bus, 1);

% Get load buses
Pd_og = model.bus(:, 3);
Qd_og = model.bus(:, 4);
load_buses_idx = find(Pd_og ~= 0 | Qd_og ~= 0);
num_load_buses = size(load_buses_idx, 1);

% Number of random samples
num_samples = 5;

% Track feasible inputs
num_feas = 0;

% Initialize array to store Pd, Qd, and costs
data = [];

% Generate sample points
p = sobolset(2 * num_load_buses); % Sobol points (uniform pseudo-random sampling in high dimensions)
X = net(p, num_samples); % Pseudo-random sampling over unit hypercube
Pd_arr = X(:, 1:num_load_buses); % Get columns corresponding to active power demand
Pd_arr = 1.5 * model.bus(load_buses_idx, 3)' .* Pd_arr; % Scale by default bus demand
Qd_arr = X(:, num_load_buses+1:end); % Get columns corresponding to reactive power demand
Qd_arr = 1.5 * model.bus(load_buses_idx, 4)' .* Qd_arr; % Scale by default bus demand

tic
for i = 1:num_samples
       
    % Re-load model
    model = eval(case_name);
  
    % Set load at buses
    model.bus(load_buses_idx, 3) = Pd_arr(i, :)'; % Pseudo-random Pd
    model.bus(load_buses_idx, 4) = Qd_arr(i, :)'; % Pesudo-random Qd

    try
        % Solve model
        results = OPF_Solver(model, settings);
        
        % Extract Pd, Qd, and cost
        Pd = model.bus(:, 3)';  % Transpose to row vector
        Qd = model.bus(:, 4)';  % Transpose to row vector
        cost = results.sdp.cost;
        
        % Append to data array
        data = [data; Pd, Qd, cost, results.feas_flag];
        
        % Display progress for feasible solutions
        fprintf('Sample %d: Cost = %.2f, Feasibility Flag = %.1f\n', i, cost, results.feas_flag);

        % Track number of feasible data points
        num_feas = num_feas + results.feas_flag;

    catch ME
        % Handle the error and skip to the next iteration
        fprintf('Sample %d: Error occurred - %s\n', i, ME.message);
        continue; % Skip this sample and move to the next one
    end
end
toc

% Save data to CSV file
fprintf('Number of feasible solutions: %d\n', num_feas)

% Headers: "Pd1", "Pd2", ..., "Pd14", "Qd1", "Qd2", ..., "Qd14", "Cost"
Pd_headers = strcat("Pd", string(1:num_buses));
Qd_headers = strcat("Qd", string(1:num_buses));
headers = [Pd_headers, Qd_headers, "Cost", "feas_flag"];
filename = sprintf('../training_data/%s_%d_samples.csv', case_name, num_samples);

% Convert to table and write to CSV
data_table = array2table(data, 'VariableNames', headers);
writetable(data_table, filename);

fprintf('Results saved to %s\n', filename);

% Clean up the temporary file after all computations are done
% delete(temp_filename);