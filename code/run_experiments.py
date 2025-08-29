import os
import csv
import datetime
import random
import math
from collections import defaultdict
import ast
import copy
import networkx as nx
import pandas as pd
import json


def load_data_from_file(file_path):

    try:
        with open(file_path, 'r') as f:
            content = f.read()
        data_list = ast.literal_eval(content)
        clients_dict = {f"c{i}": row for i, row in enumerate(data_list)}
        print(f" Successfully loaded and parsed {len(clients_dict)} data points.")
        return clients_dict
    except FileNotFoundError:
        print(f" ERROR: The file was not found at the specified path: {file_path}")
        return None
    except Exception as e:
        print(f" Error loading or parsing file: {e}")
        return None

def generate_L_k_median_heuristic(clients, facilities, k, L):

    if len(facilities) < k:
        raise ValueError(f"Cannot select k={k} facilities, pool only has {len(facilities)} options.")
    sl_facilities = dict(random.sample(list(facilities.items()), k))
    
    assignments, clusters = {}, defaultdict(list)
    for cid, c_coords in clients.items():
        nearest = min(sl_facilities.keys(), key=lambda fid: math.dist(c_coords, sl_facilities[fid]))
        assignments[cid] = nearest
        clusters[nearest].append(cid)
        
    underloaded = {fid for fid, c in clusters.items() if len(c) < L}
    overloaded = {fid for fid, c in clusters.items() if len(c) > L}
    
    for u_fac in underloaded:
        needed = L - len(clusters[u_fac])
        candidates = []
        for o_fac in overloaded:
            if len(clusters[o_fac]) > L:
                for cid in clusters[o_fac]:
                    candidates.append((math.dist(clients[cid], facilities[u_fac]), cid, o_fac))
        candidates.sort()
        stolen_count = 0
        for dist, cid, orig_fac in candidates:
            if stolen_count >= needed: break
            if len(clusters[orig_fac]) > L:
                assignments[cid] = u_fac
                clusters[orig_fac].remove(cid)
                clusters[u_fac].append(cid)
                stolen_count += 1
                
    return sl_facilities, assignments

def generate_U_k_median_heuristic(clients, facilities, k, U):

    if len(facilities) < k:
        raise ValueError(f"Cannot select k={k} facilities, pool only has {len(facilities)} options.")
    su_facilities = dict(random.sample(list(facilities.items()), k))
    
    assignments, cluster_sizes = {}, defaultdict(int)
    client_ids = list(clients.keys())
    random.shuffle(client_ids)
    
    for cid in client_ids:
        c_coords = clients[cid]
        sorted_facs = sorted(su_facilities.keys(), key=lambda fid: math.dist(c_coords, su_facilities[fid]))
        best_fac = next((fid for fid in sorted_facs if cluster_sizes[fid] < U), sorted_facs[0])
        assignments[cid] = best_fac
        cluster_sizes[best_fac] += 1
        
    return su_facilities, assignments

def generate_input_solutions(P, F, k, L, U):
    """Wrapper to generate both S_L and S_U and format the output."""
    sl_facilities, sl_assignment = generate_L_k_median_heuristic(P, F, k, L)
    su_facilities, su_assignment = generate_U_k_median_heuristic(P, F, k, U)
    return {
        "parameters": {"L": L, "U": U}, "clients": P, "sl_facilities": sl_facilities,
        "su_facilities": su_facilities, "sl_assignment": sl_assignment, "su_assignment": su_assignment
    }



def calculate_eta_mapping(sl_fac, su_fac):
    eta_mapping = {}
    for su_id, su_coords in su_fac.items():
        nearest_sl_id = min(sl_fac.keys(), key=lambda sl_id: math.dist(su_coords, sl_fac[sl_id]))
        eta_mapping[su_id] = nearest_sl_id
    return eta_mapping

def construct_g2(clients_keys, sl_assign, su_assign, eta_map, sl_fac_keys):
    star_vertices = {i for i in sl_fac_keys if i in eta_map.values()}
    G2 = nx.DiGraph()
    G2.add_nodes_from(star_vertices)
    edge_weights = defaultdict(int)
    for cid in clients_keys:
        i1, i_prime = sl_assign.get(cid), su_assign.get(cid)
        if i_prime and i1:
            i2 = eta_map.get(i_prime)
            if i1 in star_vertices and i2 in star_vertices: edge_weights[(i1, i2)] += 1
    for (i1, i2), w in edge_weights.items():
        if w > 0: G2.add_edge(i1, i2, weight=w)
    return G2

def break_cycles_fixed(g2_cyclic, sl_assignment, su_assignment, eta_mapping, clients):
    g2 = copy.deepcopy(g2_cyclic)
    sigma_hat_l = sl_assignment.copy()
    X = defaultdict(lambda: defaultdict(set))
    for cid in clients.keys():
        i1 = sl_assignment.get(cid)
        i_prime = su_assignment.get(cid)
        i2 = eta_mapping.get(i_prime)
        if i1 is not None and i2 is not None:
            X[i1][i2].add(cid)

    def edge_weight(graph, u, v):
        return graph.edges[u, v]['weight'] if graph.has_edge(u, v) and 'weight' in graph.edges[u, v] else 0

    while True:
        try:
            cycle = next(c for c in nx.simple_cycles(g2) if len(c) > 1)
        except StopIteration:
            break

        q = len(cycle)
        edges = [(cycle[i], cycle[(i + 1) % q]) for i in range(q)]
        kappa = min(edge_weight(g2, u, v) for (u, v) in edges)
        if kappa <= 0:
            zero_edges = [(u, v) for u, v, d in g2.edges(data=True) if d.get('weight', 0) == 0 and u != v]
            if zero_edges:
                g2.remove_edges_from(zero_edges)
            continue

        for (u, v) in edges:
            available = list(X[u][v])
            kappa_effective = min(kappa, len(available))
            to_move = available[:kappa_effective]
            for cid in to_move:
                sigma_hat_l[cid] = v
                X[u][v].remove(cid)
                X[v][v].add(cid)

            if g2.has_edge(u, v):
                new_w = g2.edges[u, v].get('weight', 0) - kappa_effective
                if new_w <= 0:
                    g2.remove_edge(u, v)
                else:
                    g2.edges[u, v]['weight'] = new_w

            if not g2.has_edge(v, v):
                g2.add_edge(v, v, weight=0)
            g2.edges[v, v]['weight'] = g2.edges[v, v].get('weight', 0) + kappa_effective

        zero_edges = [(u, v) for u, v, d in g2.edges(data=True) if d.get('weight', 0) == 0 and u != v]
        if zero_edges:
            g2.remove_edges_from(zero_edges)

    return g2, sigma_hat_l

def process_star(star_center, eta_inverse, all_coords, unsettled, sigma_hat, sigma_u, L, U, beta=1.0):
    spokes = eta_inverse.get(star_center, [])
    if not spokes:
        return {}, set()

    final_assignments = {}
    settled_this_step = set()
    sigma_u_inverse = defaultdict(set)
    for client, facility in sigma_u.items():
        sigma_u_inverse[facility].add(client)

    N = {spoke: unsettled.intersection(sigma_u_inverse.get(spoke, set())) for spoke in spokes}
    spokes_ordered = sorted(spokes, key=lambda s: math.dist(all_coords[s], all_coords[star_center]), reverse=True)
    
    reserved_i = set()
    last_spoke = spokes_ordered[-1]
    if len(N[last_spoke]) < L:
        num_to_reserve = L - len(N[last_spoke])
        sigma_hat_inverse = defaultdict(list)
        for client, fac in sigma_hat.items():
            sigma_hat_inverse[fac].append(client)
        candidates = [c for c in sigma_hat_inverse[star_center] if c in unsettled and c not in N[last_spoke]]
        reserved_i = set(candidates[:num_to_reserve])

    for spoke in spokes:
        N[spoke] = N[spoke] - reserved_i

    bag = set()
    for spoke in spokes_ordered[:-1]:
        bag.update(N[spoke])
        if len(bag) >= L:
            final_assignments[spoke] = set(bag)
            settled_this_step.update(bag)
            bag.clear()

    leftover_clients = bag.union(N[last_spoke]).union(reserved_i)
    if leftover_clients:
        final_assignments[star_center] = final_assignments.get(star_center, set()).union(leftover_clients)
        settled_this_step.update(leftover_clients)

    return final_assignments, settled_this_step

def calculate_clustering_cost(assignments, clients_map, all_facilities_coords):
    total_cost = 0.0
    for client_id, facility_id in assignments.items():
        if facility_id in all_facilities_coords:
            total_cost += math.dist(clients_map[client_id], all_facilities_coords[facility_id])
    return total_cost


if __name__ == '__main__':

    # Set the path to your data file
    file_path = 'file_path_here'
    
    # Define the parameter sweep values. This runs for all combinations of k_value, pool_pct_values, l_ratios, u_ratios.
    k_values = [10, 25, 50, 75, 100, 125, 150, 175, 250, 500, 750, 1000] #choose any k and any number of k
    pool_pct_values = [0.05, 0.1, 0.2, 0.25] #choose the size of facility pool of dataset. this chooses (pool_pct_values * n (size of dataset)) random points as facility pool for the generation of heuristic SL and SU.
    l_ratios = [0.5, 0.75, 0.9] #this is ratios of n/k as lower bounds. Eg.: (n=45000)/(k=10)=4500 then 0.5 here would make l = 2250
    u_ratios = [1.1, 1.25, 1.5] #this is ratios of n/k as upper bounds. Eg.: (n=45000)/(k=10)=4500 then 1.1 here would make u = 4950
    
    clients_dict = load_data_from_file(file_path)
    
    if clients_dict:
        n_clients = len(clients_dict)

        log_file_name = 'el_clustering_logs.csv' #create unique file names for different datasets here!
        log_header = [
            'timestamp', 'status', 'facility_pool_pct', 'k', 'L', 'U', 
            'cost_S_L', 'cost_S_U', 'final_cost_S_I', 
            'theoretical_bound', 'cost_as_pct_of_bound',
            'min_cluster_size', 'max_cluster_size', 
            'lower_bound_violated', 'num_lb_violations',
            'upper_bound_violation_factor',
            'all_clients_assigned'
        ]
        
        if not os.path.exists(log_file_name):
            with open(log_file_name, 'w', newline='') as f:
                csv.writer(f).writerow(log_header)

        # Parameter Sweep
        run_count = 1
        for k in k_values:
            for pool_pct in pool_pct_values:
                pool_size = int(n_clients * pool_pct)
                facility_pool_keys = random.sample(list(clients_dict.keys()), pool_size)
                facilities_dict = {key: clients_dict[key] for key in facility_pool_keys}

                for l_ratio in l_ratios:
                    for u_ratio in u_ratios:
                        L = int((n_clients / k) * l_ratio)
                        U = int((n_clients / k) * u_ratio)
                        
                        if L >= U:
                            continue

                        print(f"\n--- RUN {run_count}: k={k}, L={L}, U={U}, pool_pct={pool_pct} ---")
                        
                        try:
                            data = generate_input_solutions(clients_dict, facilities_dict, k, L, U)
                        except ValueError as e:
                            print(f"REJECTED: Heuristic failed during generation. Error: {e}")
                            log_row_base = [datetime.datetime.now().isoformat(), 'HEURISTIC_FAILED', pool_pct, k, L, U] + ['N/A'] * 11
                            with open(log_file_name, 'a', newline='') as f: csv.writer(f).writerow(log_row_base)
                            run_count += 1
                            continue

                        # EL-Clustering Pipeline
                        params = data['parameters']
                        clients, sl_fac, su_fac = data['clients'], data['sl_facilities'], data['su_facilities']
                        all_fac_coords = {**sl_fac, **su_fac, **facilities_dict}
                        
                        eta_map = calculate_eta_mapping(sl_fac, su_fac)
                        g2_cyclic = construct_g2(clients.keys(), data['sl_assignment'], data['su_assignment'], eta_map, sl_fac.keys())
                        g2_dag, sigma_hat_l = break_cycles_fixed(g2_cyclic, data['sl_assignment'], data['su_assignment'], eta_map, clients)
                        
                        try:
                            g2_for_sort = g2_dag.copy()
                            g2_for_sort.remove_edges_from(list(nx.selfloop_edges(g2_for_sort)))
                            order = list(nx.topological_sort(g2_for_sort))
                        except nx.NetworkXUnfeasible:
                            order = list(g2_dag.nodes())
                        
                        unsettled, final_solution = set(clients.keys()), {}
                        eta_inv = defaultdict(list)
                        for su, sl in eta_map.items(): eta_inv[sl].append(su)
                        
                        for center in order:
                            new_assign, settled = process_star(center, eta_inv, all_fac_coords, unsettled, sigma_hat_l, data['su_assignment'], L, U)
                            if new_assign:
                                for fac, client_set in new_assign.items():
                                    if fac not in final_solution:
                                        final_solution[fac] = set()
                                    final_solution[fac].update(client_set)
                                unsettled -= settled
                        

                        cost_S_L = calculate_clustering_cost(data['sl_assignment'], clients, all_fac_coords)
                        cost_S_U = calculate_clustering_cost(data['su_assignment'], clients, all_fac_coords)
                        
                        final_assignments_flat = {c: f for f, cs in final_solution.items() for c in cs}
                        cost_S_I = calculate_clustering_cost(final_assignments_flat, clients, all_fac_coords)
                        
                        theoretical_bound = (7 * cost_S_U + 2 * cost_S_L) if cost_S_U > 0 and cost_S_L > 0 else 0
                        pct_of_bound = (cost_S_I / theoretical_bound) * 100 if theoretical_bound > 0 else 0
                        
                        cluster_sizes = [len(c) for c in final_solution.values()]
                        min_size, max_size = (min(cluster_sizes), max(cluster_sizes)) if cluster_sizes else (0, 0)
                        
                        lower_bound_violated = 1 if min_size > 0 and min_size < L else 0
                        num_lb_violations = sum(1 for size in cluster_sizes if 0 < size < L)
                        
                        upper_bound_violation_factor = max_size / U if U > 0 else 0
                        all_clients_assigned = 1 if len(final_assignments_flat) == n_clients else 0


                        log_row = [
                            datetime.datetime.now().isoformat(), 'SUCCESS', pool_pct, k, L, U,
                            cost_S_L, cost_S_U, cost_S_I, theoretical_bound, pct_of_bound,
                            min_size, max_size, lower_bound_violated, 
                            num_lb_violations, upper_bound_violation_factor, all_clients_assigned
                        ]
                        with open(log_file_name, 'a', newline='') as f:
                            csv.writer(f).writerow(log_row)
                        
                        print(f"✅ Run {run_count} complete. Log saved to {log_file_name}.")
                        run_count += 1
        print("\n--- Parameter sweep finished ---")
