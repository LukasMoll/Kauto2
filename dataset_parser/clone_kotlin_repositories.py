import requests
import subprocess
import os
import argparse
from datetime import datetime, timedelta

def get_top_kotlin_repositories(github_token):
    """Fetches the top 100 Kotlin repositories sorted by stars."""
    url = 'https://api.github.com/search/repositories'
    headers = {'Authorization': f'token {github_token}'}
    params = {
        'q': 'language:Kotlin',
        'sort': 'stars',
        'order': 'desc',
        'per_page': 100
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()['items']
    else:
        raise Exception(f"Failed to fetch repositories: {response.content}")

def get_latest_commit(github_token, repo_full_name):
    """Fetches the latest commit from the repository."""
    url = f'https://api.github.com/repos/{repo_full_name}/commits'
    headers = {'Authorization': f'token {github_token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        latest_commit_date = response.json()[0]['commit']['committer']['date']
        return datetime.strptime(latest_commit_date, "%Y-%m-%dT%H:%M:%SZ")
    else:
        raise Exception(f"Failed to fetch commits for {repo_full_name}: {response.content}")

def should_clone(latest_commit_datetime):
    """Determines if the repository should be cloned based on the commit date."""
    now = datetime.utcnow()
    return (now - latest_commit_datetime) < timedelta(days=30)

def clone_repository(git_url, base_dir):
    """Clones a repository from the provided URL to the specified directory."""
    repo_name = git_url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join(base_dir, repo_name)
    if not os.path.exists(repo_path):
        subprocess.run(['git', 'clone', git_url, repo_path])
        print(f"Cloned {repo_name}.")
    else:
        print(f"Repository {repo_name} already exists.")

def main():
    parser = argparse.ArgumentParser(description='Clone top 100 Kotlin repositories based on stars with recent commits.')
    parser.add_argument('token', help='GitHub personal access token')
    parser.add_argument('directory', help='Directory to clone repositories into')
    args = parser.parse_args()

    os.makedirs(args.directory, exist_ok=True)
    repositories = get_top_kotlin_repositories(args.token)
    for repo in repositories:
        latest_commit = get_latest_commit(args.token, repo['full_name'])
        if should_clone(latest_commit):
            clone_repository(repo['clone_url'], args.directory)
        else:
            print(f"Skipping {repo['name']} due to older last commit.")

if __name__ == '__main__':
    main()
