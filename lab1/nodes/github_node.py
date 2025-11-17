from input_output_formats import GithubReposInfo, GraphState
import requests
from typing import List


class GitHubRepoSearcher:
    def __init__(self, github_token: str, default_max_results: int = 3):
        self.github_token = github_token
        self.default_max_results = default_max_results
        self.base_url = "https://api.github.com/search/repositories"

    def search_github_repositories(self, search_query: str, github_token: str, max_results: int) -> List[GithubReposInfo]:
        """
        Ищет репозитории на GitHub по ключевым словам
        """

        # Настраиваем заголовки с токеном
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {github_token}'
        }

        params = {
            'q': search_query,
            'per_page': min(max_results, 100),
            'sort': 'stars',  # Сортировка по звездам (самые популярные)
            'order': 'desc'
        }

        response = requests.get(
            self.base_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        search_data = response.json()
        results = []

        # Обрабатываем результаты
        for item in search_data.get('items', []):
            results.append(GithubReposInfo(name=item['full_name'], description=item.get(
                'description', ''), url=item['html_url']))

        return results

    def get_github_repos(self, state: GraphState) -> GraphState:
        repos = self.search_github_repositories(
            state['result_summary'].topic, self.github_token, self.default_max_results)
        print("\n\n\n!!!!!!!\n\n")
        # print(state['topic'])
        urls_string = '\n'.join([repo.url for repo in repos])
        print(urls_string)

        return {"urls": urls_string}

    def __call__(self, state: GraphState) -> GraphState:
        return self.get_github_repos(state)
