# Functions to get uncertainty of the parameters and states given a fitted HDP-HMM model

import numpy as np
import pandas as pd

def jaccard_similarity(a, b):
    '''Calculate Jaccard similarity between two lists'''
    A, B = set(a), set(b)
    return len(A & B) / len(A | B)


def converged_models(ihmm, nconverged, Ks=0):
    '''Get a DataFrame of the converged models'''

    niter = ihmm.iter
    converged_iter = np.arange(niter-nconverged, niter)

    df = pd.DataFrame()

    for iter in converged_iter:

        # get indices for each state
        state_sequence = ihmm.convergence['state_sequence'][iter]
        uniq_states = np.unique(state_sequence)

        state_indices = dict()
        for state_idx in uniq_states:
            my_indices = np.nonzero(state_sequence == state_idx)
            state_indices[state_idx] = list(zip(my_indices[0], my_indices[1]))

        # get state parameters
        A = np.array(ihmm.convergence['A'])[iter, :, :, uniq_states, Ks]
        invSigma = np.array(ihmm.convergence['invSigma'])[iter, :, :, uniq_states, Ks]
        
        # create temporary dataframe for this Gibbs iteration
        tmp = pd.DataFrame()
        tmp['state'] = [state for state in uniq_states]
        tmp['n_points'] = [len(state_indices[state]) for state in uniq_states]
        
        if ihmm.isotropic:
            tmp['A'] = [A[0, 0, i] for i in range(len(uniq_states))]
            tmp['invSigma'] = [invSigma[0, 0, i] for i in range(len(uniq_states))]
        else:
            tmp['A'] = [A[:, :, i] for i in range(len(uniq_states))]
            tmp['invSigma'] = [invSigma[:, :, i] for i in range(len(uniq_states))]
        
        tmp['idx'] = [state_indices[state] for state in uniq_states]
        tmp['nstates'] = len(uniq_states)
        tmp['model'] = iter

        # merge with main dataframe
        df = pd.concat([df, tmp], ignore_index=True)

    return df


def map_states(df, N):
    '''Map states between converged models using Jaccard similarity'''
    
    # only consider models with N states
    df_Nstates = df[df['nstates'] == N].copy().drop(columns=['nstates'])

    # Get all unique models
    all_models = df_Nstates['model'].unique()
    mod = df_Nstates[df_Nstates['model'] == all_models[0]].set_index('state')
    mod['new_state'] = mod['n_points'].rank(method='min', ascending=False)

    # write mod1 new_state and best_jaccard back into df_Nstates for the first model
    df_Nstates.loc[df_Nstates['model'] == all_models[0], 'new_state'] = (
        df_Nstates[df_Nstates['model'] == all_models[0]]['state'].map(mod['new_state'])
    )
    df_Nstates.loc[df_Nstates['model'] == all_models[0], 'best_jaccard'] = 1.0

    # Create a Jaccard similarity matrix for mod1
    jaccard_matrix_mod1 = pd.DataFrame(index=mod.index, columns=mod.index, dtype=float)
    for s1 in mod.index:
        for s2 in mod.index:
            jaccard_matrix_mod1.loc[s1, s2] = jaccard_similarity(mod.loc[s1, 'idx'], mod.loc[s2, 'idx'])

    # Loop through all other models and assign new_state based on Jaccard similarity to mod1
    for model_iter in all_models[1:]:
        mod_current = df_Nstates[df_Nstates['model'] == model_iter].set_index('state')
        
        # Create a mapping based on Jaccard similarity
        jaccard_matrix = pd.DataFrame(index=mod.index, columns=mod_current.index, dtype=float)
        for s1 in mod.index:
            for s2 in mod_current.index:
                jaccard_matrix.loc[s1, s2] = jaccard_similarity(mod.loc[s1, 'idx'], mod_current.loc[s2, 'idx'])
        
        # Find the best match (highest Jaccard) for each state in mod1
        state_mapping = {}
        for s1 in mod.index:
            best_match = jaccard_matrix.loc[s1].idxmax()
            best_score = jaccard_matrix.loc[s1, best_match]
            state_mapping[s1] = {'mapped_state': best_match, 'jaccard_score': best_score}
        
        # Create reverse mappings (current model state -> mod new_state, and best Jaccard score)
        current_state_mapping = {}
        best_jaccard_map = {}
        for s2 in mod_current.index:
            # best mod state for this current state
            s1 = jaccard_matrix[s2].idxmax()
            score = jaccard_matrix.loc[s1, s2]
            current_state_mapping[s2] = mod.loc[s1, 'new_state']
            best_jaccard_map[s2] = score

        df_Nstates.loc[df_Nstates['model'] == model_iter, 'new_state'] = (
            df_Nstates[df_Nstates['model'] == model_iter]['state'].map(current_state_mapping)
        )
        df_Nstates.loc[df_Nstates['model'] == model_iter, 'best_jaccard'] = (
            df_Nstates[df_Nstates['model'] == model_iter]['state'].map(best_jaccard_map)
        )

    return df_Nstates


def most_representative_model(df_Nstates):
    '''Get the most representative model with N states based on L2 distance to mean parameters'''
    
    # Compute mean parameters per new_state
    mean_params = df_Nstates.groupby('new_state')[['A', 'invSigma']].mean()

    # For each model, compute distance to means
    model_distances = {}
    for model_id in df_Nstates['model'].unique():
        model_data = df_Nstates[df_Nstates['model'] == model_id].set_index('new_state')
        
        # Compute squared distance for each state, then sum
        dist = 0
        for state_id in model_data.index:
            if state_id in mean_params.index:
                dist += (model_data.loc[state_id, 'A'].mean() - mean_params.loc[state_id, 'A'])**2
                dist += (model_data.loc[state_id, 'invSigma'].mean() - mean_params.loc[state_id, 'invSigma'])**2
        
        model_distances[model_id] = np.sqrt(dist)

    # Find model with minimum distance
    most_representative_model = min(model_distances, key=model_distances.get)
    print(f"Most representative model: {most_representative_model} (distance: {model_distances[most_representative_model]:.4f})")

    return df_Nstates[df_Nstates['model'] == most_representative_model]

