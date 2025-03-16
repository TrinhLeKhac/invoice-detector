SPECIAL_CHARACTER = r"=><„,.:;“”\/-_(){}|?"
VOCAB = r'aAáÁàÀảẢãÃạẠăĂắẮằẰẳẲẵẴặẶâÂấẤầẦẩẨẫẪậẬbBcCdDđĐeEéÉèÈẻẺẽẼẹẸêÊếẾềỀểỂễỄệỆfFgGhHiIíÍìÌỉỈĩĨịỊjJkKlLmMnNoOóÓòÒỏỎõÕọỌôÔốỐồỒổỔỗỖộỘơƠớỚờỜởỞỡỠợỢpPqQrRsStTuUúÚùÙủỦũŨụỤưƯứỨừỪửỬữỮựỰvVwWxXyYýÝỳỲỷỶỹỸỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

DIGIT_AND_MISSPELLED_CHAR = "\dOÔƠ,\.\s"
OPTIONAL_COLON = "\s*:?\s*"

# Regex to capture various datetime formats
TIME_DATE_PATTERN = r"(?:(\d{2}[:]\d{2}(?::\d{2})?)\s+)?(\d{2}[\s/-]\d{2}[\s/-]\d{4})(?:\s+(\d{2}[:]\d{2}(?::\d{2})?))?"

# CREATED_TIME_PATTERN = r"(?:(?:hoa don ban hang)[^0-9]*)?(\d{1,2}\s*[/:.-]\s*\d{1,2}\s*[/:.-]\s*\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}(?::\d{2})?\s+\d{1,2}\s*[/:.-]\s*\d{1,2}\s*[/:.-]\s*\d{4})"
CREATED_TIME_PATTERN = r"(?:(?:h[oóòỏõọôốồổỗộơớờởỡợ©]\s*[aáàảãạăắằẳẵặâấầẩẫậ]\s*[dđ]\s*[oóòỏõọôốồổỗộơớờởỡợ©]\s*[mn]\s*b[aáàảãạăắằẳẵặâấầẩẫậ]\s*[mn]\s*h[aáàảãạăắằẳẵặâấầẩẫậ]\s*[mn][qg])[^0-9]*)?(\d{1,2}\s*[/:.-]\s*\d{1,2}\s*[/:.-]\s*\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}(?::\d{2})?\s+\d{1,2}\s*[/:.-]\s*\d{1,2}\s*[/:.-]\s*\d{4})"

# SHOP_NAME_PATTERN = rf"(?:s\s*hop){OPTIONAL_COLON}(.*?)(?:hot\s*line)"
SHOP_NAME_PATTERN = rf"(?:s\s*h[oóòỏõọôốồổỗộơớờởỡợ©]p){OPTIONAL_COLON}(.*?)(?:h[oóòỏõọôốồổỗộơớờởỡợ©]t\s*l[iíìỉĩịîï][mn][eéèẻẽẹêếềểễệ])"

# HOTLINE_PATTERN = rf"(?:hot\s*line){OPTIONAL_COLON}(.*?)(?:nhan\s*vien\s*ban\s*hang)"
HOTLINE_PATTERN = rf"(?:h[oóòỏõọôốồổỗộơớờởỡợ©]t\s*l[iíìỉĩịîï][mn][eéèẻẽẹêếềểễệ]){OPTIONAL_COLON}(.*?)(?:[mn]h[aáàảãạăắằẳẵặâấầẩẫậ][mn]\s*v[iíìỉĩịîï][eéèẻẽẹêếềểễệ]n\s*b[aáàảãạăắằẳẵặâấầẩẫậ][mn]\s*h[aáàảãạăắằẳẵặâấầẩẫậ][mn][qg])"

# EMPLOYEE_NAME_PATTERN = rf"(?:nhan\s*vien\s*ban\s*hang){OPTIONAL_COLON}(.*?)(?:khach\s*hang)"
EMPLOYEE_NAME_PATTERN = rf"(?:[mn]h[aáàảãạăắằẳẵặâấầẩẫậ][mn]\s*v[iíìỉĩịîï][eéèẻẽẹêếềểễệ]n\s*b[aáàảãạăắằẳẵặâấầẩẫậ][mn]\s*h[aáàảãạăắằẳẵặâấầẩẫậ][mn][qg]){OPTIONAL_COLON}(.*?)(?:kh[aáàảãạăắằẳẵặâấầẩẫậ]ch\s*h[aáàảãạăắằẳẵặâấầẩẫậ][mn][qg])"

# CUSTOMER_NAME_PATTERN = rf"(?:khach\s*hang){OPTIONAL_COLON}(.*?)(?:s\s*d[t|f])"
CUSTOMER_NAME_PATTERN = rf"(?:kh[aáàảãạăắằẳẵặâấầẩẫậ]ch\s*h[aáàảãạăắằẳẵặâấầẩẫậ][mn][qg]){OPTIONAL_COLON}(.*?)(?:[sš]*\s*[dđp][t|f])"

# CUSTOMER_PHONE_PATTERN = rf"(?:s\s*d[t|f]){OPTIONAL_COLON}(.*?)(?:dia\s*chi)"
CUSTOMER_PHONE_PATTERN = rf"(?:[sš]*\s*[dđp][t|f]){OPTIONAL_COLON}(.*?)(?:[dđ][iíìỉĩịîï][aáàảãạăắằẳẵặâấầẩẫậ]\s*ch[iíìỉĩịîï]*)"

# ADDRESS_PATTERN = rf"(?:dia\s*chi){OPTIONAL_COLON}(.*?)(?:khu\s*vuc)"
ADDRESS_PATTERN = rf"(?:[dđ][iíìỉĩịîï][aáàảãạăắằẳẵặâấầẩẫậ]\s*ch[iíìỉĩịîï]*){OPTIONAL_COLON}(.*?)(?:kh[uúùủũụưứừửữự]\s*v[uúùủũụưứừửữự]c)"

# REGION_PATTERN = rf"(?:khu\s*vuc){OPTIONAL_COLON}(.*?)(?:thoi\s*gian\s*giao\s*hang)"
REGION_PATTERN = rf"(?:kh[uúùủũụưứừửữự]\s*v[uúùủũụưứừửữự]c){OPTIONAL_COLON}(.*?)(?:th[oóòỏõọôốồổỗộơớờởỡợ©][iíìỉĩịîï]\s*[qg][iíìỉĩịîï][aáàảãạăắằẳẵặâấầẩẫậ]n\s*[qg][iíìỉĩịîï][aáàảãạăắằẳẵặâấầẩẫậ][oóòỏõọôốồổỗộơớờởỡợ©]\s*h[aáàảãạăắằẳẵặâấầẩẫậ][mn][qg])"

# SHIPPING_TIME_PATTERN = rf"(?:thoi\s*gian\s*giao\s*hang){OPTIONAL_COLON}(.*?)(?:ten)"
SHIPPING_TIME_PATTERN = rf"(?:th[oóòỏõọôốồổỗộơớờởỡợ©][iíìỉĩịîï]\s*[qg][iíìỉĩịîï][aáàảãạăắằẳẵặâấầẩẫậ]n\s*[qg][iíìỉĩịîï][aáàảãạăắằẳẵặâấầẩẫậ][oóòỏõọôốồổỗộơớờởỡợ©]\s*h[aáàảãạăắằẳẵặâấầẩẫậ][mn][qg]){OPTIONAL_COLON}(.*?)(?:t[eéèẻẽẹêếềểễệ][mn])"

TOTAL_QUANTITY_PATTERN = rf"t[ôoòóỏọõốồổỗộơờớởỡợ]?[mn]?[qg]?\s*s[ôoòóỏọõốồổỗộơờớởỡợ]\s*l[uưừứửữự]?[oôọóòõốồổỗộơờớởỡợ]?[mn]?[qg]?{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
TOTAL_AMOUNT_PATTERN = rf"t[ôoòóỏọõốồổỗộơờớởỡợ]?[mn]?[qg]?\s*t[iíìỉĩịîï][eéèẻẽẹêếềểễệ][mn]\s*h[aáàảãạăắằẳẵặâấầẩẫậ]?[mn]?[qg]?{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
DISCOUNT_PATTERN = rf"ch[iíìỉĩịîï][eéèẻẽẹêếềểễệ]t\s*kh[aáàảãạăắằẳẵặâấầẩẫậ][uưừứửữự]\s*h[oóòỏõọôốồổỗộơớờởỡợ©]?[aáàảãạăắằẳẵặâấầẩẫậ]?\s*[đd][oóòỏõọôốồổỗộơớờởỡợ©]\S*{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
MONETARY_PATTERN = rf"t[ôoòóỏọõốồổỗộơờớởỡợ][mn]?[qg]?\s*c[oôòóỏọõốồổỗộơờớởỡợ]?[mn]?[qg]?{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"

TABLE_COLUMN_MAPPING = {
    "(?:.*t[eêềếệểễ][mn].*h[àaáạãảâấầậẩẫăắằẳẵặ][mn][qg].*h[oóòỏõọôốồổỗộơớờởỡợ©][àaáạãảâấầậẩẫăắằẳẵặ].*|"
    ".*h[àaáạãảâấầậẩẫăắằẳẵặ][mn][qg].*h[oóòỏõọôốồổỗộơớờởỡợ©][àaáạãảâấầậẩẫăắằẳẵặ].*)": "product_name",
    "(?:.*s[.,]?l.*|"
    ".*s[oóòỏõọôốồổỗộơớờởỡợ©.,].*l[ưứừửữựuúùủũụ].*[oơờớởỡợ][mn][qg].*?)": "quantity",
    "(?:.*[dđ][.,]?.*[gq][iíìỉĩị][àaáạãảâấầậẩẫăắằẳẵặ].*|"
    ".*[dđ][ôốồổỗộoóòỏõọ][mn].*[gq][iíìỉĩị][àaáạãảâấầậẩẫăắằẳẵặ].*)": "unit_price",
    "(?:.*t[.,]?t[iíìỉĩị][eêềếệểễ][mn].*|"
    ".*th[aàáạãảâấầậẩẫăắằẳẵặ][mn]h.*t[iíìỉĩị][eêềếệểễ][mn].*)": "total_price",
}
