# 1. Update src/private_llm_analytics/simulator/interactions.py
inter_path = 'src/private_llm_analytics/simulator/interactions.py'
with open(inter_path, 'r', encoding='utf-8') as f:
    i_content = f.read().replace('\r\n', '\n')

old_assert = """    n_users, n_cells = user_counts.shape
    assert n_cells == 9"""

new_assert = """    n_users, n_cells = user_counts.shape
    assert n_cells == len(providers) * len(tasks), f"Expected {len(providers) * len(tasks)} cells, got {n_cells}" """

assert old_assert in i_content, "old_assert not found in interactions.py"
i_content = i_content.replace(old_assert, new_assert)
with open(inter_path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(i_content.replace('\n', '\r\n'))
print("Updated interactions.py")

# 2. Update src/private_llm_analytics/simulator/generator.py
gen_path = 'src/private_llm_analytics/simulator/generator.py'
with open(gen_path, 'r', encoding='utf-8') as f:
    g_content = f.read().replace('\r\n', '\n')

old_gen_counts = """        # 3. Sample multinomial interaction counts across 9 cells for each user
        # Vectorized batch processing
        user_counts = np.zeros((sc.n_users, 9), dtype=np.int64)"""

new_gen_counts = """        # 3. Sample multinomial interaction counts across dynamic cells for each user
        # Vectorized batch processing
        n_cells = len(sc.providers) * len(sc.tasks)
        user_counts = np.zeros((sc.n_users, n_cells), dtype=np.int64)"""

assert old_gen_counts in g_content, "old_gen_counts not found in generator.py"
g_content = g_content.replace(old_gen_counts, new_gen_counts)
with open(gen_path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(g_content.replace('\n', '\r\n'))
print("Updated generator.py")

# 3. Update src/private_llm_analytics/simulator/users.py
users_path = 'src/private_llm_analytics/simulator/users.py'
with open(users_path, 'r', encoding='utf-8') as f:
    u_content = f.read().replace('\r\n', '\n')

old_users_body = """    assert len(providers) == 3
    assert len(tasks) == 3
    assert len(interaction_counts) == n_users

    persona_names = list(PERSONA_TASK_ALPHA.keys())
    persona_weights = [0.30, 0.30, 0.25, 0.15]

    # Assign personas
    assigned_personas_idx = rng.choice(len(persona_names), size=n_users, p=persona_weights)

    # Sample task probabilities for each user using their persona's Dirichlet distribution
    task_probs = np.zeros((n_users, 3), dtype=np.float64)
    for p_idx, p_name in enumerate(persona_names):
        mask = (assigned_personas_idx == p_idx)
        count = int(np.sum(mask))
        if count > 0:
            alpha = PERSONA_TASK_ALPHA[p_name]
            task_probs[mask] = rng.dirichlet(alpha, size=count)

    # Sample user-specific provider affinities (variation across employees)
    user_provider_affinity = rng.dirichlet(np.array([2.0, 2.0, 2.0]), size=n_users) # shape (n_users, 3)

    # Compute P(provider | task, user)
    # Combine user provider affinity with task-specific provider strengths
    # Joint probabilities: shape (n_users, 9)
    # Cell indexing: 3 providers x 3 tasks
    # Ordered as: (p_0, t_0), (p_0, t_1), (p_0, t_2), (p_1, t_0), ..., (p_2, t_2)
    # where p_i in providers, t_j in tasks
    joint_probs = np.zeros((n_users, 9), dtype=np.float64)

    base_weights_matrix = np.vstack([
        BASE_TASK_PROVIDER_WEIGHTS[tasks[0]], # task 0
        BASE_TASK_PROVIDER_WEIGHTS[tasks[1]], # task 1
        BASE_TASK_PROVIDER_WEIGHTS[tasks[2]], # task 2
    ]) # shape (3, 3) -> row=task, col=provider

    for t_idx in range(3):
        # Base weights for this task: shape (3,)
        bw = base_weights_matrix[t_idx]
        # Element-wise product of user affinity and base task weights
        cond_p = user_provider_affinity * bw # shape (n_users, 3)
        # Normalize over providers
        cond_p = cond_p / cond_p.sum(axis=1, keepdims=True)

        p_task = task_probs[:, t_idx:t_idx+1] # shape (n_users, 1)

        # For each provider p_idx:
        for p_idx in range(3):
            cell_idx = p_idx * 3 + t_idx
            joint_probs[:, cell_idx] = (p_task * cond_p[:, p_idx:p_idx+1]).ravel()

    # Normalize joint_probs to sum to 1.0 per user
    joint_probs = joint_probs / joint_probs.sum(axis=1, keepdims=True)

    pad_width = max(5, len(str(n_users)))
    user_ids = [f"U{i+1:0{pad_width}d}" for i in range(n_users)]

    return user_ids, joint_probs"""

new_users_body = """    assert len(interaction_counts) == n_users
    n_p = len(providers)
    n_t = len(tasks)
    n_cells = n_p * n_t

    if n_p == 3 and n_t == 3 and tasks == ["technical", "document", "analysis"]:
        persona_names = list(PERSONA_TASK_ALPHA.keys())
        persona_weights = [0.30, 0.30, 0.25, 0.15]
        assigned_personas_idx = rng.choice(len(persona_names), size=n_users, p=persona_weights)
        task_probs = np.zeros((n_users, 3), dtype=np.float64)
        for p_idx, p_name in enumerate(persona_names):
            mask = (assigned_personas_idx == p_idx)
            count = int(np.sum(mask))
            if count > 0:
                alpha = PERSONA_TASK_ALPHA[p_name]
                task_probs[mask] = rng.dirichlet(alpha, size=count)
        user_provider_affinity = rng.dirichlet(np.array([2.0, 2.0, 2.0]), size=n_users)
        joint_probs = np.zeros((n_users, 9), dtype=np.float64)
        base_weights_matrix = np.vstack([
            BASE_TASK_PROVIDER_WEIGHTS[tasks[0]],
            BASE_TASK_PROVIDER_WEIGHTS[tasks[1]],
            BASE_TASK_PROVIDER_WEIGHTS[tasks[2]],
        ])
        for t_idx in range(3):
            bw = base_weights_matrix[t_idx]
            cond_p = user_provider_affinity * bw
            cond_p = cond_p / cond_p.sum(axis=1, keepdims=True)
            p_task = task_probs[:, t_idx:t_idx+1]
            for p_idx in range(3):
                cell_idx = p_idx * 3 + t_idx
                joint_probs[:, cell_idx] = (p_task * cond_p[:, p_idx:p_idx+1]).ravel()
        joint_probs = joint_probs / joint_probs.sum(axis=1, keepdims=True)
    else:
        # Dynamic long-tail Zipfian task distribution across 10-15+ tasks
        zipf_weights = np.array([1.0 / ((i + 1) ** 0.85) for i in range(n_t)])
        base_alpha = 15.0 * (zipf_weights / zipf_weights.sum())
        base_alpha = np.maximum(base_alpha, 0.2)
        task_probs = rng.dirichlet(base_alpha, size=n_users)

        prov_alpha = np.ones(n_p) * 2.0
        user_provider_affinity = rng.dirichlet(prov_alpha, size=n_users)

        joint_probs = np.zeros((n_users, n_cells), dtype=np.float64)
        for t_idx in range(n_t):
            # Base provider preference with slight variation across tasks
            t_hash = hash(tasks[t_idx]) % 1000 / 1000.0
            p_weights = np.array([1.0 + 0.3 * np.sin(t_hash * (i + 1)) for i in range(n_p)])
            p_weights = p_weights / p_weights.sum()

            cond_p = user_provider_affinity * p_weights
            cond_p = cond_p / cond_p.sum(axis=1, keepdims=True)

            p_task = task_probs[:, t_idx:t_idx+1]
            for p_idx in range(n_p):
                cell_idx = p_idx * n_t + t_idx
                joint_probs[:, cell_idx] = (p_task * cond_p[:, p_idx:p_idx+1]).ravel()

        joint_probs = joint_probs / joint_probs.sum(axis=1, keepdims=True)

    pad_width = max(5, len(str(n_users)))
    user_ids = [f"U{i+1:0{pad_width}d}" for i in range(n_users)]
    return user_ids, joint_probs"""

assert old_users_body in u_content, "old_users_body not found in users.py"
u_content = u_content.replace(old_users_body, new_users_body)
with open(users_path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(u_content.replace('\n', '\r\n'))
print("Updated users.py")
