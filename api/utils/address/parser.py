from unidecode import unidecode
from utils.address.helper import *


def parse_address(
    input_address, db_provinces, db_districts, db_wards, special_ending, debug=False
):

    # Preprocess the address
    normalized_address = remove_punctuation(input_address, ADDRESS_PUNCTUATIONS)
    normalized_address = add_space_separator(normalized_address)

    address = clean_full_address(normalized_address) + ","

    founded_province = ""
    founded_district = ""
    founded_ward = ""

    choose_word = ""
    largest_index = -1

    # Extract Province/City
    for province, province_data in db_provinces.items():
        for word in province_data["words"]:
            # Select the closest component (Province, District, Ward) from the right of the address
            last_index = address.rfind(word)
            # If two components have the same index, prioritize the longer string
            if (last_index > largest_index) or (
                (last_index == largest_index)
                and (largest_index > -1)
                and (len(word) > len(choose_word))
            ):
                founded_province = province
                choose_word = word
                largest_index = last_index

    # If a Province/City is found, remove it from the address (only the last occurrence)
    if founded_province:
        address = replace_last_occurrences(address, choose_word, "")

    if address.endswith("Thanh Pho ,"):
        address = address.replace("Thanh Pho ,", "")

    choose_word = ""
    largest_index = -1

    # Extract District after extracting Province/City
    if founded_province:
        for district in db_provinces[founded_province]["district"]:
            for word in db_districts[district]["words"]:
                reg_word = re.compile(rf"{word}{special_ending}")
                last_index = last_index_of_regex(address, reg_word)
                if (last_index > largest_index) or (
                    (last_index == largest_index)
                    and (largest_index > -1)
                    and (len(word) > len(choose_word))
                ):
                    founded_district = district
                    choose_word = word
                    largest_index = last_index
    else:
        # Extract District when Province/City is not present in the address
        # (Inferring Province/City from the District)
        for district, district_data in db_districts.items():
            for word in district_data["words"]:
                reg_word = re.compile(rf"{word}")
                last_index = last_index_of_regex(address, reg_word)
                if (last_index > largest_index) or (
                    (last_index == largest_index)
                    and (largest_index > -1)
                    and (len(word) > len(choose_word))
                ):
                    founded_district = district
                    choose_word = word
                    largest_index = last_index

    if founded_district:
        address = replace_last_occurrences(address, choose_word, "")
        if not founded_province:
            province_id = db_districts[founded_district]["province"]
            founded_province = next(
                (
                    key
                    for key, value in db_provinces.items()
                    if value["id"] == province_id
                ),
                "",
            )

    largest_index = -1
    choose_word = ""

    # Extract Ward/Commune
    if founded_district:
        for ward in db_districts[founded_district]["ward"]:
            for word in db_wards[ward]["words"]:
                reg_word = re.compile(rf"{word}{special_ending}")
                last_index = last_index_of_regex(address, reg_word)
                if (last_index > largest_index) or (
                    (last_index == largest_index)
                    and (largest_index > -1)
                    and (len(word) > len(choose_word))
                ):
                    founded_ward = ward
                    choose_word = word
                    largest_index = last_index
        if founded_ward:
            address = replace_last_occurrences(address, choose_word, "")

    # Map results to Province, District, and Ward names
    founded_province = db_provinces.get(founded_province, {}).get("name", "")
    founded_district = db_districts.get(founded_district, {}).get("name", "")
    founded_ward = db_wards.get(founded_ward, {}).get("name", "")

    # Remove redundant parts after extracting province/city, district, ward from NO-ACCENT address
    tmp_address = remove_abundant_part(address)

    # Convert the normalized address to a no-accent version
    no_accent_normalized_address = unidecode(normalized_address)

    if debug:
        print(tmp_address)
        print(normalized_address)
        print(no_accent_normalized_address)

    # Find address lv4 (street address) in the original address
    address_lv4 = matching_and_find_substring(
        normalized_address, no_accent_normalized_address, tmp_address
    )

    # Correct misspelled words
    address_lv4_no_accent = unidecode(address_lv4)
    address_lv4 = correct_misspelled_words(address_lv4, address_lv4_no_accent)
    if debug:
        print(address_lv4)

    # Filter out empty values before joining
    address_parts = [address_lv4, founded_ward, founded_district, founded_province]
    modified_address = ", ".join(filter(None, address_parts))
    modified_address = re.sub(r"\s+", " ", modified_address).strip()

    return modified_address
