import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.address import *
import unittest


class TestInitCapWords(unittest.TestCase):
    def test_normal_case(self):
        self.assertEqual(init_cap_words("hello world"), "Hello World")

    def test_already_capitalized(self):
        self.assertEqual(init_cap_words("Hello World"), "Hello World")

    def test_mixed_case(self):
        self.assertEqual(init_cap_words("hElLo WoRlD"), "Hello World")

    def test_extra_spaces(self):
        self.assertEqual(init_cap_words("   hello    world  "), "Hello World")

    def test_empty_string(self):
        self.assertEqual(init_cap_words(""), "")

    def test_single_word(self):
        self.assertEqual(init_cap_words("python"), "Python")

    def test_numbers_and_symbols(self):
        self.assertEqual(init_cap_words("123 test case!"), "123 Test Case!")

    def test_non_string_input(self):
        with self.assertRaises(ValueError):
            init_cap_words(123)

    def test_special_characters(self):
        self.assertEqual(
            init_cap_words("hello-world"), "Hello-world"
        )  # Keeps hyphenated words intact


class TestRemoveAccent(unittest.TestCase):
    def test_normal_vietnamese_text(self):
        self.assertEqual(remove_accent("Tôi đang ăn phở"), "Toi dang an pho")
        self.assertEqual(
            remove_accent("Hà Nội là thủ đô của Việt Nam"),
            "Ha Noi la thu do cua Viet Nam",
        )

    def test_vietnamese_with_uppercase(self):
        self.assertEqual(remove_accent("ĐẠI HỌC BÁCH KHOA"), "DAI HOC BACH KHOA")
        self.assertEqual(
            remove_accent("THÀNH PHỐ HỒ CHÍ MINH"), "THANH PHO HO CHI MINH"
        )

    def test_vietnamese_with_mixed_case(self):
        self.assertEqual(
            remove_accent("Hà Nội và TP. Hồ Chí Minh"), "Ha Noi va TP. Ho Chi Minh"
        )
        self.assertEqual(remove_accent("Cần Thơ - Tây Đô"), "Can Tho - Tay Do")

    def test_text_without_accents(self):
        self.assertEqual(remove_accent("Hello world"), "Hello world")
        self.assertEqual(remove_accent("Python is fun"), "Python is fun")

    def test_numbers_and_symbols(self):
        self.assertEqual(remove_accent("1234!@#$%^&*()"), "1234!@#$%^&*()")
        self.assertEqual(remove_accent("Hà Nội 1000 năm"), "Ha Noi 1000 nam")

    def test_empty_string(self):
        self.assertEqual(remove_accent(""), "")

    def test_non_string_input(self):
        with self.assertRaises(ValueError):
            remove_accent(123)
        with self.assertRaises(ValueError):
            remove_accent(None)
        with self.assertRaises(ValueError):
            remove_accent(["Hà Nội"])


class TestCleanDashAddress(unittest.TestCase):
    def setUp(self):
        self.dict_norm_city_dash = {
            "Ba Ria - Vung Tau": [
                "Ba Ria Vung Tau",
                "Ba Ria-Vung Tau",
                "Brvt",
                "Br - Vt",
            ],
            "Phan Rang - Thap Cham": ["Phan Rang Thap Cham"],
            "Thua Thien Hue": ["Thua Thien - Hue"],
            "Ho Chi Minh": ["Sai Gon", "Tphcm", "Hcm", "Sg"],
            "Da Nang": ["Quang Nam-Da Nang", "Quang Nam - Da Nang"],
        }

    def test_standard_replacements(self):
        self.assertEqual(
            clean_dash_address("I live in Brvt", self.dict_norm_city_dash),
            "I live in Ba Ria - Vung Tau",
        )
        self.assertEqual(
            clean_dash_address("Visiting Sai Gon", self.dict_norm_city_dash),
            "Visiting Ho Chi Minh",
        )
        self.assertEqual(
            clean_dash_address("Exploring Quang Nam-Da Nang", self.dict_norm_city_dash),
            "Exploring Da Nang",
        )

    def test_no_replacement_needed(self):
        self.assertEqual(
            clean_dash_address("I live in Hanoi", self.dict_norm_city_dash),
            "I live in Hanoi",
        )

    def test_mixed_case_replacement(self):
        self.assertEqual(
            clean_dash_address("Enjoying hcm nightlife", self.dict_norm_city_dash),
            "Enjoying Ho Chi Minh nightlife",
        )

    def test_partial_word_not_replaced(self):
        self.assertEqual(
            clean_dash_address("Brvton Beach is great", self.dict_norm_city_dash),
            "Brvton Beach is great",
        )  # pattern = r"(" + "|".join(map(re.escape, synonyms)) + r")"

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            clean_dash_address(123, self.dict_norm_city_dash)
        with self.assertRaises(ValueError):
            clean_dash_address("Sai Gon", "invalid_dict")


class TestCleanAbbrevAddress(unittest.TestCase):
    def setUp(self):
        self.dict_norm_abbrev = {
            "Tp ": ["Tp.", "Tp:"],
            "Tt ": ["Tt.", "Tt:"],
            "Q ": ["Q.", "Q:"],
            "H ": ["H.", "H:"],
            "X ": ["X.", "X:"],
            "P ": ["P.", "P:"],
        }

    def test_standard_replacement(self):
        self.assertEqual(
            clean_abbrev_address("Tp. Ho Chi Minh", self.dict_norm_abbrev),
            "Tp Ho Chi Minh",
        )
        self.assertEqual(clean_abbrev_address("Q. 1", self.dict_norm_abbrev), "Q 1")
        self.assertEqual(
            clean_abbrev_address("P. Ben Nghe", self.dict_norm_abbrev), "P Ben Nghe"
        )

    def test_no_change_needed(self):
        self.assertEqual(
            clean_abbrev_address("Ho Chi Minh City", self.dict_norm_abbrev),
            "Ho Chi Minh City",
        )

    def test_case_insensitive_replacement(self):
        self.assertEqual(
            clean_abbrev_address("tp. hanoi", self.dict_norm_abbrev), "Tp hanoi"
        )
        self.assertEqual(clean_abbrev_address("q:5", self.dict_norm_abbrev), "Q 5")

    def test_partial_word_not_replaced(self):
        self.assertEqual(
            clean_abbrev_address("Tpbank is a bank", self.dict_norm_abbrev),
            "Tpbank is a bank",
        )

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            clean_abbrev_address(123, self.dict_norm_abbrev)
        with self.assertRaises(ValueError):
            clean_abbrev_address("Tp. Ho Chi Minh", "not a dictionary")


class TestCleanDigitDistrict(unittest.TestCase):
    def test_standard_cases(self):
        self.assertEqual(clean_digit_district("quan 1"), "Q1")
        self.assertEqual(clean_digit_district("Quan 10"), "Q10")
        self.assertEqual(clean_digit_district("q 5"), "Q5")
        self.assertEqual(clean_digit_district("Q 07"), "Q7")

    def test_leading_zeros(self):
        self.assertEqual(clean_digit_district("Q01"), "Q1")
        self.assertEqual(clean_digit_district("Q002"), "Q2")
        self.assertEqual(clean_digit_district("quan 09"), "Q9")

    def test_mixed_text(self):
        self.assertEqual(clean_digit_district("I live in quan 3"), "I live in Q3")
        self.assertEqual(clean_digit_district("This is Q05"), "This is Q5")
        self.assertEqual(
            clean_digit_district("q 10 is a big district"), "Q10 is a big district"
        )

    def test_no_changes_needed(self):
        self.assertEqual(clean_digit_district("District Q7"), "District Q7")
        self.assertEqual(clean_digit_district("Hello World"), "Hello World")
        self.assertEqual(clean_digit_district("Q1"), "Q1")

    def test_case_insensitivity(self):
        self.assertEqual(clean_digit_district("QuAn 6"), "Q6")
        self.assertEqual(clean_digit_district("Q 08"), "Q8")
        self.assertEqual(clean_digit_district("QUAN 4"), "Q4")


class TestCleanDigitWard(unittest.TestCase):
    def test_standard_cases(self):
        self.assertEqual(clean_digit_ward("phuong 1"), "P1")
        self.assertEqual(clean_digit_ward("Phuong 10"), "P10")
        self.assertEqual(clean_digit_ward("p 5"), "P5")
        self.assertEqual(clean_digit_ward("P 12"), "P12")

    def test_f_to_p_conversion(self):
        self.assertEqual(clean_digit_ward("F3"), "P3")
        self.assertEqual(clean_digit_ward("F09"), "P9")

    def test_leading_zeros(self):
        self.assertEqual(clean_digit_ward("P01"), "P1")
        self.assertEqual(clean_digit_ward("P002"), "P2")
        self.assertEqual(clean_digit_ward("P0003"), "P3")

    def test_mixed_text(self):
        self.assertEqual(clean_digit_ward("123 Phuong 07, Quan 3"), "123 P7, Quan 3")
        self.assertEqual(clean_digit_ward("F1, District 1"), "P1, District 1")
        self.assertEqual(clean_digit_ward("P02 near P10"), "P2 near P10")

    def test_no_changes_needed(self):
        self.assertEqual(
            clean_digit_ward("This is a normal sentence."), "This is a normal sentence."
        )
        self.assertEqual(clean_digit_ward("P1 already correct"), "P1 already correct")

    def test_case_insensitivity(self):
        self.assertEqual(clean_digit_ward("PHUONG 4"), "P4")
        self.assertEqual(clean_digit_ward("phuong 6"), "P6")
        self.assertEqual(clean_digit_ward("f7"), "P7")


class TestRemoveSpareSpace(unittest.TestCase):
    def test_normal_spaces(self):
        self.assertEqual(remove_spare_space("Hello  World"), "Hello World")

    def test_leading_trailing_spaces(self):
        self.assertEqual(remove_spare_space("  Hello World  "), "Hello World")

    def test_multiple_spaces(self):
        self.assertEqual(
            remove_spare_space("Hello    World   Again"), "Hello World Again"
        )

    def test_only_spaces(self):
        self.assertEqual(remove_spare_space("      "), "")

    def test_no_change(self):
        self.assertEqual(remove_spare_space("Hello World"), "Hello World")

    def test_mixed_spaces(self):
        self.assertEqual(
            remove_spare_space("  This   is   a   test  "), "This is a test"
        )


class TestRemovePunctuation(unittest.TestCase):
    def test_remove_common_punctuation(self):
        self.assertEqual(remove_punctuation("Hello, world!", [",", "!"]), "Hello world")

    def test_remove_multiple_punctuation_types(self):
        self.assertEqual(
            remove_punctuation("Hello... world!!!", [".", "!"]), "Hello world"
        )

    def test_no_punctuation_in_text(self):
        self.assertEqual(remove_punctuation("Hello world", [",", "."]), "Hello world")

    def test_empty_punctuation_list(self):
        self.assertEqual(remove_punctuation("Hello, world!", []), "Hello, world!")

    def test_only_punctuation_characters(self):
        self.assertEqual(remove_punctuation("...,,,!!!", [".", ",", "!"]), "")

    def test_mixed_spacing_after_removal(self):
        self.assertEqual(
            remove_punctuation("Hello , world !", [",", "!"]), "Hello world"
        )

    def test_special_regex_characters(self):
        self.assertEqual(
            remove_punctuation("Hello (world)?", ["(", ")", "?"]), "Hello world"
        )


class TestAddSpaceSeparator(unittest.TestCase):
    def test_add_space_after_comma(self):
        self.assertEqual(add_space_separator("Hello,world"), "Hello, World")
        self.assertEqual(add_space_separator("Hello ,world"), "Hello, World")

    def test_replace_special_characters_with_space(self):
        self.assertEqual(add_space_separator("Hello.World"), "Hello World")
        self.assertEqual(add_space_separator("Hello-World"), "Hello World")
        self.assertEqual(add_space_separator("Hello_World"), "Hello World")

    def test_normalize_multiple_spaces(self):
        self.assertEqual(add_space_separator("Hello    world"), "Hello World")
        self.assertEqual(add_space_separator("Hello -  world"), "Hello World")

    def test_capitalize_first_letter_of_words(self):
        self.assertEqual(add_space_separator("hello world"), "Hello World")
        self.assertEqual(add_space_separator("hello,world"), "Hello, World")

    def test_mixed_cases_and_symbols(self):
        self.assertEqual(
            add_space_separator("hello.world,how-are_you"), "Hello World, How Are You"
        )
        self.assertEqual(add_space_separator("HELLO_WORLD"), "Hello World")

    def test_empty_string(self):
        self.assertEqual(add_space_separator(""), "")


class TestLastIndexOfRegex(unittest.TestCase):
    def test_single_match(self):
        self.assertEqual(last_index_of_regex("hello world", "world"), 6)
        self.assertEqual(last_index_of_regex("test 123 test", "123"), 5)

    def test_multiple_matches(self):
        self.assertEqual(last_index_of_regex("abc abc abc", "abc"), 8)
        self.assertEqual(last_index_of_regex("hello hello hello", "hello"), 12)

    def test_no_match(self):
        self.assertEqual(last_index_of_regex("hello world", "Python"), -1)
        self.assertEqual(last_index_of_regex("test test", "123"), -1)

    def test_special_characters(self):
        self.assertEqual(
            last_index_of_regex("a.b.c.d", r"\."), 5
        )  # Tìm vị trí của dấu '.'
        self.assertEqual(
            last_index_of_regex("abc$def$ghi", r"\$"), 7
        )  # Tìm vị trí ký tự '$'

    def test_numeric_patterns(self):
        self.assertEqual(last_index_of_regex("123 456 789 123", "123"), 12)
        self.assertEqual(last_index_of_regex("000111000", "111"), 3)

    def test_empty_string(self):
        self.assertEqual(last_index_of_regex("", "hello"), -1)
        self.assertEqual(last_index_of_regex("", ""), -1)


class TestReplaceLastOccurrences(unittest.TestCase):
    def test_single_occurrence(self):
        self.assertEqual(
            replace_last_occurrences("hello world", "world", "Python"), "hello Python"
        )
        self.assertEqual(replace_last_occurrences("test 123", "123", "456"), "test 456")

    def test_multiple_occurrences(self):
        self.assertEqual(
            replace_last_occurrences("abc abc abc", "abc", "xyz"), "abc abc xyz"
        )
        self.assertEqual(
            replace_last_occurrences("one two three two", "two", "2"), "one two three 2"
        )

    def test_no_occurrence(self):
        self.assertEqual(
            replace_last_occurrences("hello world", "Python", "Java"), "hello world"
        )
        self.assertEqual(
            replace_last_occurrences("test test", "123", "456"), "test test"
        )

    def test_replace_with_empty_string(self):
        self.assertEqual(replace_last_occurrences("abc abc abc", "abc", ""), "abc abc ")
        self.assertEqual(
            replace_last_occurrences("hello world world", "world", ""), "hello world "
        )

    def test_replace_empty_substring(self):
        self.assertEqual(
            replace_last_occurrences("hello world", "", "X"), "hello world"
        )

    def test_empty_string_cases(self):
        self.assertEqual(replace_last_occurrences("", "test", "new"), "")
        self.assertEqual(replace_last_occurrences("test", "", "new"), "test")


if __name__ == "__main__":
    unittest.main()
