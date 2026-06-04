import re
def obfuscate_url(match):
    url = match.group(0)
    protocol_match = re.match(r'https?://', url)
    if not protocol_match:
        return url
    protocol = protocol_match.group(0)
    if len(url) < len(protocol) + 2:
        return url
    last_two = url[-2:]
    return f"{protocol}***{last_two}"

def obfuscate_username(username):
    if not username: return "Unknown"
    if len(username) <= 2: return username
    return f"{username[0]}***{username[-1]}"

def truncate_text(text, limit):
    if not isinstance(text, str): return ""
    if len(text) <= limit: return text
    return text[:limit] + "..."

def obfuscate_mention(match):
    mention = match.group(0)
    if len(mention) <= 2:
        return mention
    return mention[0] + mention[1] + '***' + mention[-1]


def obfuscate_retweet(match):
    full = match.group(0)
    user_match = re.search(r'@(\w+)', full)
    if not user_match:
        return full
    username = user_match.group(1)
    if len(username) <= 2:
        obf_user = username
    else:
        obf_user = username[0] + '***' + username[-1]
    return f"RT @{obf_user}:"



def get_first_k_tokens(text, k):
    """
    Extracts the first k tokens from a text string.

    :param text: The input text string.
    :param k: The number of tokens to extract.
    :return: The first k tokens of the text string.
    """
    # Split the text into tokens based on whitespace
    tokens = text.split()
    output = " ".join(tokens[:k])

    # Return the first k tokens
    return output

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list



