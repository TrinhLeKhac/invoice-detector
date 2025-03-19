import itertools
import re
import unicodedata

from unidecode import unidecode

from utils.text.regex import TIME_DATE_PATTERN


def clean_text_before_unidecode(text):
    """
    Cleans the text by removing:
    - Special characters that may expand (e.g., œ -> oe)
    - Emojis
    - Invisible whitespace characters (\u200b, \u00a0)
    - Soft hyphen (\u00ad)
    """

    # # Remove special characters (keeps only letters, digits, and spaces)
    # text = re.sub(r'[^A-Za-zÀ-ỹ0-9\s]', '', text)

    # Remove emojis using regex (matches all Unicode emoji ranges)
    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]",
        "",
        text,
    )

    # Remove invisible whitespace characters and soft hyphen
    text = re.sub(r"[\u200b\u00A0\u00AD]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_information(target, no_accent_target, pattern, direct=False):
    """
    Extracts the desired information from `target` using regex on `no_accent_target`.

    This function applies the regex pattern to `no_accent_target`, then retrieves
    the corresponding substring from `target` based on the matched index range.

    Args:
        target (str): The original text (with accents).
        no_accent_target (str): The text without accents, used for regex matching.
        pattern (str): The regex pattern for extraction.

    Returns:
        str: Extracted information if found, else an empty string.
    """

    if direct:
        match = re.search(pattern, target, re.IGNORECASE)

        if match:
            return match.group(1)  # Capture the first group
        return ""
    else:
        match = re.search(pattern, no_accent_target, re.IGNORECASE)

        if match:
            # Get the start and end indices of the captured group in `no_accent_target`
            start_idx = match.start(1)
            end_idx = match.end(1)

            # Extract the corresponding substring from the original `target`
            information = target[start_idx:end_idx]

            return information
        return ""


def extract_and_normalize_phone_numbers(text):
    """
    Extracts and normalizes phone numbers from a given text.

    This function searches for phone numbers that:
    - Start with "+084", "+84", or "0"
    - Are followed by exactly **9 or 10 digits**

    The extracted numbers are normalized by:
    - Removing "+084" and "+84", replacing them with "0"
    - Ensuring all phone numbers are returned in a standard "0XXXXXXXXX(X)" format

    Args:
        text (str): The input string containing phone numbers.

    Returns:
        list: A list of normalized phone numbers in "0XXXXXXXXX(X)" format.
    # Remove numbers and special characters, keeping only letters and spaces
    """

    # Matches phone numbers starting with +084, +84, or 0 and followed by 9 or 10 digits
    pattern = r"\b(?:\+084|\+84|0)(\d{9,10})\b"

    # Get list of matched numbers
    matches = re.findall(pattern, text)

    # Replace +84, +084 with 0
    normalized_numbers = ["0" + num for num in matches]

    return normalized_numbers


def remove_consecutive_duplicate_tone_marks(word):
    """
    Remove one of two consecutive characters if they have the same base character after removing tone marks.

    Parameters:
    - word (str): Input word with potential consecutive duplicate tone-marked characters.

    Returns:
    - str: The corrected word with consecutive duplicate tone-marked characters removed.
    """

    result = []
    prev_base = ""

    for char in word:
        base_char = unidecode(char).lower()  # Get the character without diacritics
        if (
            base_char != prev_base
        ):  # Keep the character if different from the previous one
            result.append(char)
        prev_base = base_char  # Update previous character

    return "".join(result)


def remove_spaces_in_brackets(text):
    return re.sub(
        r"(\(|\{|\[)\s+|\s+(\)|\}|\])", lambda m: m.group(1) or m.group(2), text
    )


def extract_name(text):
    """
    Extracts and cleans a name from a given text by removing unwanted characters.

    This function performs the following steps:
    - Removes special characters**, keeping only letters (both English and accented characters) and spaces.
    - Normalizes spaces**, ensuring there are no extra or trailing spaces.

    Args:
        text (str): The input string containing a name.

    Returns:
        str: A cleaned name with only valid characters and properly formatted spacing.
    """

    # Remove special characters, keeping only letters (including accents) and spaces
    cleaned_text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", "", text)

    # Remove spaces in bracket if has
    cleaned_text = remove_spaces_in_brackets(cleaned_text)

    # Normalize spaces (convert multiple spaces to a single space)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # Remove one of two consecutive characters if they have the same base character after removing tone marks
    # Eg. LÊE THUUUU NGÂÂN => LÊ THU NGÂN
    cleaned_text = remove_consecutive_duplicate_tone_marks(cleaned_text)

    return cleaned_text


def normalize_product_name(product_name: str, tokens: dict) -> str:
    """
    Normalize a product name by replacing words with the highest-weighted tokens.

    Parameters:
    - product_name (str): The input product name (which may contain words with or without diacritics).
    - tokens (dict): A dictionary where keys are token words (with diacritics) and values are their respective weights.

    Returns:
    - str: The normalized product name with the most relevant tokens.
    """

    def get_best_match(phrase, token_map):
        """
        Find the best matching token from the token_map for a given phrase.

        Parameters:
        - phrase (str): The input phrase to search for.
        - token_map (dict): A dictionary mapping non-accented phrases to their best matching accented tokens.

        Returns:
        - str or None: The best matching token if found, otherwise None.
        """
        phrase_no_accent = unidecode(phrase).lower()
        return token_map.get(phrase_no_accent, None)

    # Create a mapping from non-accented phrases to the highest-weighted accented tokens
    token_map = {}
    for token in tokens:
        token_no_accent = unidecode(token).lower()
        if (
            token_no_accent not in token_map
            or len(token.split()) > len(token_map[token_no_accent].split())
            or tokens[token] > tokens[token_map[token_no_accent]]
        ):
            token_map[token_no_accent] = token

    # Remove | when detecting misrecognized characters from table borders
    product_name = re.sub(r"[|']", "", product_name)
    product_name = re.sub(r"\s+", " ", product_name).strip()

    # Normalize product name by removing consecutive duplicate tone marks
    product_name = remove_consecutive_duplicate_tone_marks(product_name)

    # Preserve special characters like parentheses in the final output
    words = re.split(
        r"(\s+|[()\[\]{}])", product_name
    )  # Split while keeping delimiters
    words = [w for w in words if w != " "]
    words = [w for w in words if w != ""]
    print("Original words: ", words)

    i = 0
    while i < len(words):
        if words[i].strip() in "()[]{}":
            i += 1
            continue

        max_match = None
        max_length = 0

        # Check all possible n-grams starting at position i
        for n in range(len(words) - i, 0, -1):
            phrase = " ".join([re.sub(r"[()\[\]{}]", "", w) for w in words[i : i + n]])
            print("Phrase: ", phrase)
            match = get_best_match(phrase, token_map)
            if match:
                max_match = match
                max_length = n
                print("Max match: ", max_match)
                break

        # Replace words with the best match if found
        if max_match:
            if words[i][0] in "()[]{}":  # Preserve parentheses around replaced word
                words[i : i + max_length] = [words[i][0] + max_match + words[i][-1]]
            else:
                words[i : i + max_length] = [max_match]
            print("Modified works: ", words)
        i += max_length if max_match else 1

    normalized_product_name = remove_spaces_in_brackets(" ".join(words).title())
    return normalized_product_name


def normalize_name_by_weight(
    name: str, first_names: dict, middle_names: dict, last_names: dict, debug=False
) -> str:
    """
    Normalize a Vietnamese name by correcting misspellings, capitalizing words,
    and rearranging components in the correct order: first_names (Họ) - middle_names (Chữ lót) - last_names (Tên).

    Args:
        name (str): The input name to be normalized.
        first_names (dict): A dictionary of valid first names (Họ) with priority scores.
        middle_names (dict): A dictionary of valid middle names (Chữ lót) with priority scores.
        last_names (dict): A dictionary of valid last names (Tên) with priority scores.

    Returns:
        str: The normalized full name or an empty string if no valid name is found.
    """

    def find_best_match(word, valid_dict):
        """Find the best match for a word in the given dictionary by comparing its non-accented version."""
        candidates = [
            vn_name
            for vn_name in valid_dict
            if unidecode(vn_name).lower() == unidecode(word).lower()
        ]
        if candidates:
            return max(
                candidates, key=lambda x: valid_dict[x]
            )  # Choose the one with the highest priority
        return None

    words = name.split()
    words = [
        word.capitalize() for word in words
    ]  # Capitalize the first letter of each word

    # Remove invalid words and correct misspellings
    valid_words = []
    for word in words:
        if word in first_names or word in middle_names or word in last_names:
            valid_words.append(word)  # Keep valid words
        else:
            match = find_best_match(word, {**first_names, **middle_names, **last_names})
            if match:
                valid_words.append(match)  # Replace misspelled words with best matches

    if not valid_words:
        return ""

    if debug:
        print("Valid words: ", valid_words)
    first, middle, last = "", [], ""

    # **Step 1: Identify the first_name (Họ)**
    # Find exact matches in the first_names dictionary
    found_first_names = [word for word in valid_words if word in first_names]
    if debug:
        print("found_first_names: ", found_first_names)

    # Find potential first_name by checking for misspelled versions in first_name
    found_first_matches = {
        match: word
        for word in valid_words
        if word not in found_first_names
        and (match := find_best_match(word, first_names))
    }
    if debug:
        print("found_first_matches: ", found_first_matches)

    # Combine both exact and corrected first_name candidates
    all_first_candidates = found_first_names + list(found_first_matches.keys())
    if debug:
        print("all_matches: ", all_first_candidates)

    # Select the first_name with the highest priority score
    if all_first_candidates:
        first = max(all_first_candidates, key=lambda x: first_names[x])
        if debug:
            print("Họ: ", first)
        if first in found_first_names:
            if debug:
                print("Remove from first_names")
            valid_words.remove(first)  # Remove the selected first_name
        else:
            if debug:
                print("Remove from first_matches")
            valid_words.remove(
                found_first_matches[first]
            )  # Remove the misspelled version

    # **Step 2: Identify the last_name (Tên)**
    # Find exact matches for last_name (including cases where it appears in last_name)
    found_last_names = [
        word for word in reversed(valid_words) if word in {**first_names, **last_names}
    ]
    if debug:
        print("found_last_names: ", found_last_names)

    # Select the first name with the highest priority
    if found_last_names:
        last = max(found_last_names, key=lambda x: {**first_names, **last_names}.get(x))

        # 🔹 **Check if a better version exists with correct accents**
        unaccented_last = unidecode(last).lower()
        better_last = max(
            (name for name in last_names if unidecode(name).lower() == unaccented_last),
            key=lambda x: last_names[x],
            default=last,
        )
        if better_last != last and last_names[better_last] > last_names[last]:
            last = better_last  # Replace with the correct accented version

        if debug:
            print("Tên: ", last)

        # 🔥 ** Remove original word
        original_last = next(
            (
                word
                for word in valid_words
                if unidecode(word).lower() == unaccented_last
            ),
            last,
        )
        if original_last in valid_words:
            valid_words.remove(original_last)
    else:
        if debug:
            print("Tên else")
        for word in reversed(valid_words):
            match = find_best_match(word, {**first_names, **last_names})
            if match:
                last = match
                if debug:
                    print("Tên: ", last)
                valid_words.remove(word)
                break

    # **Step 3: Remaining words are treated as Middle Name(s) (Chữ lót)**
    middle = valid_words

    return " ".join(itertools.chain([first], middle, [last])).strip()


def extract_address(text):
    """
    Extracts and cleans an address by removing unwanted characters.

    This function ensures that:
    Only valid address characters are kept**, including:
       - Letters (both English and accented characters)
       - Numbers (0-9)
       - Spaces
       - Commas (,), slashes (/), and hyphens (-) (commonly used in addresses)
    Extra spaces are normalized** to ensure proper formatting.

    Args:
        text (str): The input string containing an address.

    Returns:
        str: A cleaned address with only valid characters and properly formatted spacing.
    """

    # Keep only letters, numbers, spaces, commas, slashes, and hyphens
    cleaned_text = re.sub(r"[^a-zA-ZÀ-ỹ0-9,\-\/\s]", "", text)

    # Normalize spaces (convert multiple spaces to a single space)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def normalize_number(currency_str):
    """
    Normalize a currency string with potential OCR errors.

    Functionality:
    - Replace commonly misrecognized characters ('O', 'Ô', 'Ơ', 'C') with '0'.
    - Remove all non-numeric characters.
    - Convert the cleaned string into an integer.

    Parameters:
        currency_str (str): The currency string that may contain OCR errors.

    Returns:
        int: The extracted number if valid.
        0: If the input is invalid or cannot be converted.
    """
    if not currency_str:
        return -1  # Handle empty string case

    # Replace 'O', 'Ô', 'Ơ' (common OCR errors) with '0'
    currency_str = re.sub(r"[oôơc]", "0", currency_str, flags=re.IGNORECASE)

    # Remove all non-numeric characters except digits
    cleaned_number = re.sub(r"[^\d]", "", currency_str)

    # Check if it's a valid number format
    try:
        number = int(cleaned_number)
        return number
    except ValueError:
        return -1  # Return -1 for invalid inputs


def validate_and_fill_amounts(total_amount, discount, monetary):
    """
    Validates and fills missing values for total_amount, discount, and monetary
    based on the condition: total_amount = discount + monetary.

    Conditions:
    - total_amount > 0
    - discount <= total_amount
    - If discount > total_amount, set discount = total_amount
    - If any of the three values are missing, compute it using the formula.
    """

    # Ensure discount, monetary is within valid range

    if monetary == -1:
        monetary = total_amount - discount
    elif discount == -1:
        discount = total_amount - monetary
    elif total_amount == -1:
        total_amount = discount + monetary

    if (discount > total_amount) or (monetary > total_amount):
        total_amount = discount + monetary

    return total_amount, discount, monetary


def normalize_datetime(text):
    match = re.search(TIME_DATE_PATTERN, text)
    if match:
        time_part_1 = match.group(1)  # Case where the time appears before the date
        date_part = match.group(2)  # Date in DD/MM/YYYY or DD-MM-YYYY format
        time_part_2 = match.group(3)  # Case where the time appears after the date

        # Select the correct time part (prioritizing the one after the date)
        time_part = time_part_2 if time_part_2 else time_part_1
        time_part = time_part.strip() if time_part else "00:00:00"

        # Normalize the date separator (replace multiple spaces, slashes, or dashes with "/")
        date_part = re.sub(r"[\s/-]+", "/", date_part.strip())

        # Normalize the time separator (replace multiple spaces or colons with ":")
        time_part = re.sub(r"[\s:]+", ":", time_part.strip())

        # If only HH:MM is provided, append ":00" to complete the time format
        if len(time_part) == 5:
            time_part += ":00"

        datetime_str = f"{date_part} {time_part}"

        # Convert to datetime object
        # try:
        #     dt_obj = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
        #     return dt_obj.strftime("%d/%m/%Y %H:%M:%S")
        # except ValueError:
        #     return f"Invalid format: {datetime_str}"

        return datetime_str
    return ""


# def test_normalize_datetime():
#     # Test cases
#     test_cases = [
#         "03/01/2025 18:10",
#         "03/01/2025 18:10:10",
#         "18:10:10 03/01/2025",
#         "03-01-2025 18:10",
#         "03-01-2025 18:10:10",
#         "18:10:10 03-01-2025",
#         "03-01-2025",
#         "03/01/2025",
#         "03  /  01  /  2025   18:10",
#         "18:10:10   03 - 01 - 2025",
#     ]
#     # Run test cases
#     for test in test_cases:
#         print(f"Input: {test}\nOutput: {normalize_datetime(test)}\n")
