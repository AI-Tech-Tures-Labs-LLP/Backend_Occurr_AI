def sql_response_gen(query: str) -> str:
    # In real implementation, convert query → SQL → run on database
    # For demo, just mock the output
    if "average" in query and "weekend" in query:
        return "Your average heart rate during weekends is 76 bpm."
    elif "highest" in query:
        return "Your highest recorded heart rate is 179 bpm."
    elif "lowest" in query:
        return "Your lowest recorded heart rate is 45 bpm."
    else:
        return "Simulated SQL response: your data has been processed."
