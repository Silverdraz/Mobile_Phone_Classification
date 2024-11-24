from prefect import flow
from prefect.runner.storage import GitRepository

if __name__ == "__main__":
    flow.from_source(
        source="http://github.com/Silverdraz/Mobile_Phone_Classification.git",
        entrypoint=r"src\test.py:main_flow",
    ).deploy(name="mobile_phone_1",
        work_pool_name="mobile_phone_classification",
        cron="* * * * *")
    # main_flow.deploy(
    #     name="mobile_phone_1",
    #     work_pool_name="mobile_phone_classification",
    #     push=True,
    #     cron="* * * * *",
    # )
    # my_flow = flow.from_source(
    #     source="https://github.com/Silverdraz/Mobile_Phone_Classification.git",
    #     entrypoint="./src/test.py:main_flow",
    # )
    # my_flow()
    # gh_storage = GitHubRepository(
    # repository="https://github.com/Silverdraz/Mobile_Phone_Classification.git"
    # )
    # github_repo = GitRepository(
    # url="https://github.com/Silverdraz/Mobile_Phone_Classification.git"
    # )

    # github_repo.destination
    