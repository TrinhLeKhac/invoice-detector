import pandas as pd
from underthesea import word_tokenize
from unidecode import unidecode

VIETNAMESE_LETTERS_REGEX = r"[^A-Za-zàáạảãâầấậẩẫăằắặẳẵÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪèéẹẻẽêềếệểễÈÉẸẺẼÊỀẾỆỂỄòóọỏõôồốộổỗơờớợởỡÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠìíịỉĩÌÍỊỈĨùúụủũưừứựửữƯỪỨỰỬỮÙÚỤỦŨỳýỵỷỹỲÝỴỶỸđĐ\s]"


def clean_text(text):
    """Chuẩn hóa văn bản: .title(), bỏ ký tự đặc biệt, bỏ khoảng trắng dư thừa"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", "", text)  # Giữ lại chữ cái và khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()  # Loại bỏ khoảng trắng thừa
    return text.title()  # Viết hoa chữ cái đầu mỗi từ


def safe_tokenize(text):
    if (
        isinstance(text, str) and text.strip()
    ):  # Kiểm tra có phải chuỗi không và không rỗng
        try:
            return word_tokenize(text)
        except Exception as e:
            print(f"Lỗi khi tokenize: {text} - {e}")  # Debug nếu cần
            return []
    return []


def tokenize():
    product_df = pd.read_excel("data/product.xlsx")
    product_df = product_df[["Tên hàng"]].drop_duplicates()
    product_df.columns = ["product_name"]

    target_df = product_df.reset_index(drop=True).copy()
    target_df["product_name"] = target_df["product_name"].str.title()
    target_df["cleaned_product_name"] = target_df["product_name"].replace(
        VIETNAMESE_LETTERS_REGEX, "", regex=True
    )
    target_df["cleaned_product_name"] = (
        target_df["cleaned_product_name"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    target_df["token"] = target_df["cleaned_product_name"].apply(safe_tokenize)
    target_exploded = target_df.explode("token")
    # print(target_exploded.head())
    word_count_dict = target_exploded["token"].value_counts()
    # print(word_count_dict)
    token = pd.DataFrame(word_count_dict)["count"].to_dict()
    print(token)
    return token


if __name__ == "__main__":
    tokenize()
