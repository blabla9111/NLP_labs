from langchain.tools import tool

@tool(description="Asks the user to enter a new query if the previous one does not contain a scientific topic")
def get_new_query_from_user() -> str:
    user_query = input("В вашем предыдущем запросе не удалось обнаружить тему для научного поиска.\nПожалуйста, введите ваш запрос: ")
    return user_query.strip()