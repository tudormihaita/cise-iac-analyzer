import pandas as pd

from tfimpact.collection.select_repos import mark_fork_duplicates
from tfimpact.processing.gitio import root_commit


def test_fork_duplicates_keep_highest_starred_first():
    df = pd.DataFrame(
        {
            "full_name": ["orig/a", "other/b", "fork/a", "broken/c"],
            "stars": [100, 50, 10, 5],
            "root_sha": ["r1", "r2", "r1", None],
        }
    )
    assert mark_fork_duplicates(df).tolist() == [False, False, True, False]


def test_root_commit_is_oldest_parentless(make_git_repo):
    repo = make_git_repo()
    first = repo.commit({"a.tf": 'variable "x" {}\n'})
    repo.commit({"b.tf": 'variable "y" {}\n'})
    assert root_commit(repo.path) == first
