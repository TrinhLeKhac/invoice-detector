SPECIAL_CHARACTER = r"=><„,.:;“”\/-_(){}|?"
VOCAB = r'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

DIGIT_AND_MISSPELLED_CHAR = "\dOÔƠ,\.\s"
OPTIONAL_COLON = "\s*:?\s*"

# Regex to capture various datetime formats
TIME_DATE_PATTERN = r"(?:(\d{2}[:]\d{2}(?::\d{2})?)\s+)?(\d{2}[\s/-]\d{2}[\s/-]\d{4})(?:\s+(\d{2}[:]\d{2}(?::\d{2})?))?"

# CREATED_TIME_PATTERN = r"(?:(?:hoa don ban hang)[^0-9]*)?(\d{1,2}\s*[/:.-]\s*\d{1,2}\s*[/:.-]\s*\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}(?::\d{2})?\s+\d{1,2}\s*[/:.-]\s*\d{1,2}\s*[/:.-]\s*\d{4})"
CREATED_TIME_PATTERN = r"(?:(?:h[oòóỏọõôốồổỗộơờớởỡợ]\s*[aàáãạâấầẩẫậăắằẳẵặ]\s*[dđ]\s*[oòóỏọõôốồổỗộơờớởỡợ]\s*[mn]\s*b[aàáãạâấầẩẫậăắằẳẵặ]\s*[mn]\s*h[aàáãạâấầẩẫậăắằẳẵặ]\s*[mn][qg])[^0-9]*)?(\d{1,2}\s*[/:.-]\s*\d{1,2}\s*[/:.-]\s*\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}(?::\d{2})?\s+\d{1,2}\s*[/:.-]\s*\d{1,2}\s*[/:.-]\s*\d{4})"

# SHOP_NAME_PATTERN = rf"(?:s\s*hop){OPTIONAL_COLON}(.*?)(?:hot\s*line)"
SHOP_NAME_PATTERN = rf"(?:s\s*h[oòóỏọõôốồổỗộơờớởỡợ]p){OPTIONAL_COLON}(.*?)(?:h[oòóỏọõôốồổỗộơờớởỡợ]t\s*l[iìíĩịîï][mn][êeéèẽẹêềếểễệ])"

# HOTLINE_PATTERN = rf"(?:hot\s*line){OPTIONAL_COLON}(.*?)(?:nhan\s*vien\s*ban\s*hang)"
HOTLINE_PATTERN = rf"(?:h[oòóỏọõôốồổỗộơờớởỡợ]t\s*l[iìíĩịîï][mn][êeéèẽẹêềếểễệ]){OPTIONAL_COLON}(.*?)(?:[mn]h[aàáãạâấầẩẫậăắằẳẵặ][mn]\s*v[iìíĩịîï][êeéèẽẹêềếểễệ]n\s*b[aàáãạâấầẩẫậăắằẳẵặ][mn]\s*h[aàáãạâấầẩẫậăắằẳẵặ][mn][qg])"

# EMPLOYEE_NAME_PATTERN = rf"(?:nhan\s*vien\s*ban\s*hang){OPTIONAL_COLON}(.*?)(?:khach\s*hang)"
EMPLOYEE_NAME_PATTERN = rf"(?:[mn]h[aàáãạâấầẩẫậăắằẳẵặ][mn]\s*v[iìíĩịîï][êeéèẽẹêềếểễệ]n\s*b[aàáãạâấầẩẫậăắằẳẵặ][mn]\s*h[aàáãạâấầẩẫậăắằẳẵặ][mn][qg]){OPTIONAL_COLON}(.*?)(?:kh[aàáãạâấầẩẫậăắằẳẵặ]ch\s*h[aàáãạâấầẩẫậăắằẳẵặ][mn][qg])"

# CUSTOMER_NAME_PATTERN = rf"(?:khach\s*hang){OPTIONAL_COLON}(.*?)(?:s\s*d[t|f])"
CUSTOMER_NAME_PATTERN = rf"(?:kh[aàáãạâấầẩẫậăắằẳẵặ]ch\s*h[aàáãạâấầẩẫậăắằẳẵặ][mn][qg]){OPTIONAL_COLON}(.*?)(?:s\s*[dđ][t|f])"

# CUSTOMER_PHONE_PATTERN = rf"(?:s\s*d[t|f]){OPTIONAL_COLON}(.*?)(?:dia\s*chi)"
CUSTOMER_PHONE_PATTERN = rf"(?:s\s*[dđ][t|f]){OPTIONAL_COLON}(.*?)(?:[dđ][iìíĩịîï][aàáãạâấầẩẫậăắằẳẵặ]\s*ch[iìíĩịîï])"

# ADDRESS_PATTERN = rf"(?:dia\s*chi){OPTIONAL_COLON}(.*?)(?:khu\s*vuc)"
ADDRESS_PATTERN = rf"(?:[dđ][iìíĩịîï][aàáãạâấầẩẫậăắằẳẵặ]\s*ch[iìíĩịîï]){OPTIONAL_COLON}(.*?)(?:kh[uùúủũụưừứửữự]\s*v[uùúủũụưừứửữự]c)"

# REGION_PATTERN = rf"(?:khu\s*vuc){OPTIONAL_COLON}(.*?)(?:thoi\s*gian\s*giao\s*hang)"
REGION_PATTERN = rf"(?:kh[uùúủũụưừứửữự]\s*v[uùúủũụưừứửữự]c){OPTIONAL_COLON}(.*?)(?:th[oòóỏọõôốồổỗộơờớởỡợ][iìíĩịîï]\s*[qg][iìíĩịîï][aàáãạâấầẩẫậăắằẳẵặ]n\s*[qg][iìíĩịîï][aàáãạâấầẩẫậăắằẳẵặ][oòóỏọõôốồổỗộơờớởỡợ]\s*h[aàáãạâấầẩẫậăắằẳẵặ][mn][qg])"

# SHIPPING_TIME_PATTERN = rf"(?:thoi\s*gian\s*giao\s*hang){OPTIONAL_COLON}(.*?)(?:ten)"
SHIPPING_TIME_PATTERN = rf"(?:th[oòóỏọõôốồổỗộơờớởỡợ][iìíĩịîï]\s*[qg][iìíĩịîï][aàáãạâấầẩẫậăắằẳẵặ]n\s*[qg][iìíĩịîï][aàáãạâấầẩẫậăắằẳẵặ][oòóỏọõôốồổỗộơờớởỡợ]\s*h[aàáãạâấầẩẫậăắằẳẵặ][mn][qg]){OPTIONAL_COLON}(.*?)(?:t[êeéèẽẹêềếểễệ][mn])"

TOTAL_QUANTITY_PATTERN = rf"t[ôoòóỏọõốồổỗộơờớởỡợ]ng\s*s[ôoòóỏọõốồổỗộơờớởỡợ]\s*l[uưừứửữự][oôọóòõốồổỗộơờớởỡợ]ng{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
TOTAL_AMOUNT_PATTERN = rf"t[ôoòóỏọõốồổỗộơờớởỡợ]ng\s*t[iìíĩịîï][êeéèẽẹêềếểễệ]n\s*h[aàáãạâấầẩẫậăắằẳẵặ]ng{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
DISCOUNT_PATTERN = rf"ch[iìíĩịîï][êeéèẽẹêềếểễệ]t\s*kh[aàáãạâấầẩẫậăắằẳẵặ][uưừứửữự]\s*h[oòóỏọõôốồổỗộơờớởỡợ][aàáãạâấầẩẫậăắằẳẵặ]\s*[đd][oòóỏọõôốồổỗộơờớởỡợ]n{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
MONETARY_PATTERN = rf"t[ôoòóỏọõốồổỗộơờớởỡợ]ng\s*c[oôòóỏọõốồổỗộơờớởỡợ]ng{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"

TABLE_COLUMN_MAPPING = {
    "(?:.*t[eêềếệểễ][mn].*h[àaáạãảâấầậẩẫăắằẳẵặ][mn][qg].*h[oóòỏõọôốồổỗộơớờởỡợ][àaáạãảâấầậẩẫăắằẳẵặ].*|"
    ".*h[àaáạãảâấầậẩẫăắằẳẵặ][mn][qg].*h[oóòỏõọôốồổỗộơớờởỡợ][àaáạãảâấầậẩẫăắằẳẵặ].*)": "product_name",
    "(?:.*s[.,]?l.*|"
    ".*s[oóòỏõọôốồổỗộơớờởỡợ.,].*l[ưứừửữựuúùủũụ].*[oơờớởỡợ][mn][qg].*?)": "quantity",
    "(?:.*[dđ][.,]?.*[gq][iíìỉĩị][àaáạãảâấầậẩẫăắằẳẵặ].*|"
    ".*[dđ][ôốồổỗộoóòỏõọ][mn].*[gq][iíìỉĩị][àaáạãảâấầậẩẫăắằẳẵặ].*)": "unit_price",
    "(?:.*t[.,]?t[iíìỉĩị][eêềếệểễ][mn].*|"
    ".*th[aàáạãảâấầậẩẫăắằẳẵặ][mn]h.*t[iíìỉĩị][eêềếệểễ][mn].*)": "total_price",
}
