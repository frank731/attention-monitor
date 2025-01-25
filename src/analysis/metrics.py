def calculate_attention_percentage(attention_data):
    """
    Calculate the percentage of individuals paying attention.

    Parameters:
    attention_data (list): A list of boolean values indicating attention status of individuals.

    Returns:
    float: The percentage of individuals paying attention.
    """
    if not attention_data:
        return 0.0
    return (sum(attention_data) / len(attention_data)) * 100

def summarize_attention(attention_data):
    """
    Summarize attention data.

    Parameters:
    attention_data (list): A list of boolean values indicating attention status of individuals.

    Returns:
    dict: A summary containing total individuals and attention percentage.
    """
    total_individuals = len(attention_data)
    attention_percentage = calculate_attention_percentage(attention_data)
    return {
        'total_individuals': total_individuals,
        'attention_percentage': attention_percentage
    }