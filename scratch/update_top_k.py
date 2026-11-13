with open('src/private_llm_analytics/metrics/top_k_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """    def compute_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        return self.profile.compute_exact(user_matrix)

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        return self.profile.build_user_contributions(user_matrix)

    def get_sensitivity(self, mechanism_type: str = "gaussian") -> float:
        return self.profile.sensitivity()

    def post_process(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
    ) -> Dict[str, Any]:
        return self.profile.post_process(dp_vector, user_matrix)

    def release_dp(
        self,
        user_matrix: UserMatrix,
        mechanism: DPMechanism,
        rng: Optional[np.random.Generator] = None,
    ) -> DPReleaseResult:
        contributions = self.build_user_contributions(user_matrix)
        bounded_sum = np.sum(contributions, axis=0)
        noisy_vec = mechanism.add_noise(bounded_sum, rng=rng)
        post_processed = self.post_process(noisy_vec, user_matrix)"""

new_block = """    def compute_exact(
        self,
        user_matrix: UserMatrix,
        allowed_providers: Optional[List[str]] = None,
    ) -> np.ndarray:
        return self.profile.compute_exact(user_matrix, allowed_providers=allowed_providers)

    def build_user_contributions(
        self,
        user_matrix: UserMatrix,
        allowed_providers: Optional[List[str]] = None,
    ) -> np.ndarray:
        return self.profile.build_user_contributions(user_matrix, allowed_providers=allowed_providers)

    def get_sensitivity(self, mechanism_type: str = "gaussian") -> float:
        return self.profile.sensitivity()

    def post_process(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.profile.post_process(dp_vector, user_matrix, k=k)

    def release_dp(
        self,
        user_matrix: UserMatrix,
        mechanism: DPMechanism,
        rng: Optional[np.random.Generator] = None,
        allowed_providers: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> DPReleaseResult:
        contributions = self.build_user_contributions(user_matrix, allowed_providers=allowed_providers)
        bounded_sum = np.sum(contributions, axis=0)
        noisy_vec = mechanism.add_noise(bounded_sum, rng=rng)
        post_processed = self.post_process(noisy_vec, user_matrix, k=k)"""

# Handle CRLF vs LF
content_norm = content.replace("\r\n", "\n")
assert old_block in content_norm, "old_block not found in content_norm"
content_norm = content_norm.replace(old_block, new_block)
with open('src/private_llm_analytics/metrics/top_k_tasks.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(content_norm.replace('\n', '\r\n'))
print("Successfully updated metrics/top_k_tasks.py")
