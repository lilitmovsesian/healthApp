from datetime import datetime

from src.database.session import SessionLocal
from src.database.models import BloodTestResultModel, UserModel
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("mcp_server")


def serialize_blood_test_result(result: BloodTestResultModel) -> dict[str, str | int | None]:
    return {
        "id": getattr(result, "id", None),
        "test_date": getattr(result, "test_date", None),
        "test_name": getattr(result, "test_name", None),
        "value": getattr(result, "value", None),
        "unit": getattr(result, "unit", None),
        "document_id": getattr(result, "document_id", None),
        "user_id": getattr(result, "user_id", None),
    }


@mcp.tool()
def search_latest_blood_test_results(test_name: str, limit: int):
    """Search for the latest blood test results by exact test name.

    Queries the database for blood test results matching the exact test name,
    ordered by test date in descending order to retrieve the most recent results.

    Args:
        test_name: The exact name of the blood test to search for.
        limit: Maximum number of results to return.

    Returns:
        A list of plain dictionaries with the most recent blood test values.
    """
    with SessionLocal() as session:
        results = (
            session.query(BloodTestResultModel)
            .filter(BloodTestResultModel.test_name == test_name)
            .order_by(BloodTestResultModel.test_date.desc())
            .limit(limit)
            .all()
        )
        return [serialize_blood_test_result(result) for result in results]

@mcp.tool()
def get_age() -> int:
    """Get the age of a user.

    Args:
        runtime: The tool runtime context.

    Returns:
        The age of the user as an integer.
    """
    with SessionLocal() as session:
        user = session.query(UserModel).filter(UserModel.id == 1).first()
        if user and user.date_of_birth:
            today = datetime.date.today()
            age = today.year - user.date_of_birth.year - ((today.month, today.day) < (user.date_of_birth.month, user.date_of_birth.day))
            return age
        else:
            return None

@mcp.tool()
def get_sex() -> str:
    """Get the sex of a user."""
    with SessionLocal() as session:
        user = session.query(UserModel).filter(UserModel.id == 1).first()
        if user:
            return user.sex
        else:
            return None

@mcp.tool()
def search_common_labs_interpretation(test_name: str) -> str:
    """Search for common lab interpretations by exact test name.

    Args:
        test_name: The exact name of the blood test to search for.

    Returns:
        A string with the reference values and common lab interpretation for the specified test.
    """


if __name__ == "__main__":
    mcp.run(transport="stdio")