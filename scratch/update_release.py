path = 'src/private_llm_analytics/release/release.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read().replace('\r\n', '\n')

# 1. Scoped user_matrix
target1 = """        # 6. Pre-validation and Plan generation with budget_scope_id (Section A)"""
repl1 = """        # Resolve recipient provider scope
        allowed_providers = None if "*" in policy.providers else list(policy.providers)
        if allowed_providers is not None and query.metric == "top_k_tasks":
            effective_user_matrix = user_matrix.filter_providers(allowed_providers)
        else:
            effective_user_matrix = user_matrix

        # 6. Pre-validation and Plan generation with budget_scope_id (Section A)"""

assert target1 in content, "target1 not found"
content = content.replace(target1, repl1, 1)

# 2. Cache hit 1
target2 = """                return self._finalize_output(
                    release_result=committed,
                    query=query,
                    policy=policy,
                    user_matrix=user_matrix,
                    include_exact=include_exact,
                    cache_hit=True,
                    release_key=plan.release_key,
                    budget_scope_id=budget_scope_id,
                    accountant_mode=effective_accountant_mode,
                    effective_epsilon_before=eps_fresh,
                    effective_epsilon_after=eps_fresh,
                    best_alpha=alpha_fresh,
                    release_count=count_fresh,
                    candidate_rdp=None,
                )"""

repl2 = """                return self._finalize_output(
                    release_result=committed,
                    query=query,
                    policy=policy,
                    user_matrix=effective_user_matrix,
                    include_exact=include_exact,
                    cache_hit=True,
                    release_key=plan.release_key,
                    release_id=getattr(committed, "release_id", None) or plan.release_key,
                    budget_scope_id=budget_scope_id,
                    accountant_mode=effective_accountant_mode,
                    effective_epsilon_before=eps_fresh,
                    effective_epsilon_after=eps_fresh,
                    best_alpha=alpha_fresh,
                    release_count=count_fresh,
                    candidate_rdp=None,
                )"""

assert target2 in content, "target2 not found"
content = content.replace(target2, repl2, 1)

# 3. Cache hit 2
target3 = """            return self._finalize_output(
                release_result=cached_res,
                query=query,
                policy=policy,
                user_matrix=user_matrix,
                include_exact=include_exact,
                cache_hit=True,
                release_key=plan.release_key,
                budget_scope_id=budget_scope_id,
                accountant_mode=effective_accountant_mode,
                effective_epsilon_before=eps_fresh,
                effective_epsilon_after=eps_fresh,
                best_alpha=alpha_fresh,
                release_count=count_fresh,
                candidate_rdp=None,
            )"""

repl3 = """            return self._finalize_output(
                release_result=cached_res,
                query=query,
                policy=policy,
                user_matrix=effective_user_matrix,
                include_exact=include_exact,
                cache_hit=True,
                release_key=plan.release_key,
                release_id=release_id or plan.release_key,
                budget_scope_id=budget_scope_id,
                accountant_mode=effective_accountant_mode,
                effective_epsilon_before=eps_fresh,
                effective_epsilon_after=eps_fresh,
                best_alpha=alpha_fresh,
                release_count=count_fresh,
                candidate_rdp=None,
            )"""

assert target3 in content, "target3 not found"
content = content.replace(target3, repl3, 1)

# 4. metric.release_dp
target4 = """            metric = self.catalog.get(plan.metric)
            release_result = metric.release_dp(user_matrix, mechanism, rng=execution_rng)"""

repl4 = """            metric = self.catalog.get(plan.metric)
            query_k = getattr(query, "k", None)
            if plan.metric == "top_k_tasks":
                release_result = metric.release_dp(effective_user_matrix, mechanism, rng=execution_rng, k=query_k)
            else:
                release_result = metric.release_dp(effective_user_matrix, mechanism, rng=execution_rng)"""

assert target4 in content, "target4 not found"
content = content.replace(target4, repl4, 1)

# 5. finalize call after commit
target5 = """        return self._finalize_output(
            release_result=release_result,
            query=query,
            policy=policy,
            user_matrix=user_matrix,
            include_exact=include_exact,
            cache_hit=False,
            release_key=plan.release_key,
            budget_scope_id=budget_scope_id,
            accountant_mode=effective_accountant_mode,
            effective_epsilon_before=eps_before,
            effective_epsilon_after=eps_after,
            best_alpha=best_alpha_after,
            release_count=count_after,
            candidate_rdp=candidate_rdp,
        )"""

repl5 = """        return self._finalize_output(
            release_result=release_result,
            query=query,
            policy=policy,
            user_matrix=effective_user_matrix,
            include_exact=include_exact,
            cache_hit=False,
            release_key=plan.release_key,
            release_id=release_id,
            budget_scope_id=budget_scope_id,
            accountant_mode=effective_accountant_mode,
            effective_epsilon_before=eps_before,
            effective_epsilon_after=eps_after,
            best_alpha=best_alpha_after,
            release_count=count_after,
            candidate_rdp=candidate_rdp,
        )"""

assert target5 in content, "target5 not found"
content = content.replace(target5, repl5, 1)

# 6. _finalize_output signature & body
target6 = """        candidate_rdp: Optional[Dict[Any, Any]] = None,
    ) -> Dict[str, Any]:
        exact_data = None"""

repl6 = """        candidate_rdp: Optional[Dict[Any, Any]] = None,
        release_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        exact_data = None"""

assert target6 in content, "target6 not found"
content = content.replace(target6, repl6, 1)

target7 = """        output["cache_hit"] = cache_hit
        output["release_key"] = release_key"""

repl7 = """        output["cache_hit"] = cache_hit
        output["release_key"] = release_key
        output["release_id"] = str(release_id) if release_id else release_key
        query_k = getattr(query, "k", None)
        if query.metric == "top_k_tasks" and query_k is not None:
            metric_obj = self.catalog.get(query.metric)
            if hasattr(metric_obj, "post_process"):
                output["results"] = metric_obj.post_process(release_result.dp_vector, user_matrix, k=query_k)"""

assert target7 in content, "target7 not found"
content = content.replace(target7, repl7, 1)

with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(content.replace('\n', '\r\n'))
print("Successfully updated release.py")
