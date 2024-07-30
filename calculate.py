import math
import re
import numpy as np
import tensorflow as tf
import pickle
import cv2
import sympy


model_path = 'models/major_model_new.h5'
model_trig_path = 'models/major_model_trig.h5'
label_encoder_path = 'models/label_encoder.pkl'
label_encoder_trig_path = 'models/label_encoder_trig.pkl'
scaler_path = 'models/scaler.pkl'

model = tf.keras.models.load_model(model_path)
model_trig = tf.keras.models.load_model(model_trig_path)

with open(label_encoder_path, 'rb') as le_file:
    label_encoder = pickle.load(le_file)

with open(label_encoder_trig_path, 'rb') as le_file:
    label_encoder_trig = pickle.load(le_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def is_power(c1, c2):
    delta_x = c2[0][0] - c1[0][0]
    delta_y = c2[0][1] - c1[0][1]
    angle_deg = math.degrees(math.atan2(delta_y, delta_x))
    return 20 <= angle_deg <= 70

def combine_adjacent_elements(contour_predict):
    numbers = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero'}
    combined_elements = []
    i = 0
    n = len(contour_predict)
    while i < n:
        current_element = contour_predict[i]
        if current_element[1] in numbers:
            j = i + 1
            while j < n and contour_predict[j][1] in numbers and not is_power(current_element, contour_predict[j]):
                current_element = (
                    ((current_element[0][0] + contour_predict[j][0][0]) / 2,
                     (current_element[0][1] + contour_predict[j][0][1]) / 2),
                    current_element[1] + ' ' + contour_predict[j][1]
                )
                j += 1
            combined_elements.append(current_element)
            i = j
        else:
            combined_elements.append(current_element)
            i += 1
    return combined_elements

def combine_power(contour_predict):
    combined = []
    i = 0
    while i < len(contour_predict):
        j = i + 1
        if j < len(contour_predict) and is_power(contour_predict[i], contour_predict[j]):
            base_coords = contour_predict[i][0]
            base_string = contour_predict[i][1]
            power_string = contour_predict[j][1]
            new_string = f"{base_string} power {power_string}"
            combined.append((base_coords, new_string))
            i = j + 1
        else:
            combined.append(contour_predict[i])
            i += 1
    return combined

def replace_words_and_operators(contour_predict, input_str):
    for i in contour_predict:
        input_str += i[1]

    replacements = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'plus': '+', 'minus': '-', 'mul': '*', 'power': '^', 'ob': '(',
        'cb': ')'
    }

    for word, replacement in replacements.items():
        input_str = input_str.replace(word, replacement)

    input_str = input_str.replace(" ", "")
    return input_str

def replace_root(input_str):
    pattern = r'root(\d+)'
    matches = re.finditer(pattern, input_str)
    updated_str = input_str

    for match in matches:
        root_value = int(match.group(1))
        square_root_value = math.sqrt(root_value)
        updated_str = updated_str[:match.start()] + str(int(square_root_value)) + updated_str[match.end():]

    return updated_str

def calculate_expression(expression):
    try:
        expression = expression.replace('^', '**')
        words = expression.split()
        for i, word in enumerate(words):
            if word.isalpha():
                try:
                    num = sympy.sympify(word)
                    words[i] = str(num)
                except (ValueError, sympy.SympifyError):
                    raise ValueError(f"Invalid word: {word}")

        expression = ' '.join(words)

        if not re.match(r'^[\d\(\)\+\-\*\/\^\s]+$', expression):
            raise ValueError("Invalid characters in the expression")

        result = eval(expression, {'math': math})

        if isinstance(result, (int, float)):
            return result
        else:
            raise ValueError("Invalid result")
    except Exception as e:
        return None

def format_eq(exp):
    formatted_exp = ''
    i = 0
    while i < len(exp):
        if exp[i:i+4] == 'root':
            formatted_exp += '√'
            i += 4
        elif exp[i:i+2] == '^':
            formatted_exp += f'<sup>{exp[i+2]}</sup>'
            i += 3
        else:
            formatted_exp += exp[i]
            i += 1
    return formatted_exp

def parse_equation(equation):
    equation = equation.replace(' ', '')
    parts = equation.split('=')
    if len(parts) != 2:
        raise ValueError("Invalid equation format")

    lhs, rhs = parts
    lhs_coeffs = [0, 0, 0] 
    lhs_terms = lhs.split('+') + lhs.split('-')

    for term in lhs_terms:
        if 'x^3' in term:
            coeff = term.split('x^3')[0]
            if coeff == '':
                coeff = '1'
            lhs_coeffs[0] = int(coeff)
        elif 'x^2' in term:
            coeff = term.split('x^2')[0]
            if coeff == '':
                coeff = '1'
            lhs_coeffs[1] = int(coeff)
        elif 'x' in term:
            coeff = term.split('x')[0]
            if coeff == '' or coeff == '-':
                coeff = '1'
            lhs_coeffs[2] = int(coeff)

    rhs_value = int(rhs)
    return lhs_coeffs, rhs_value

def calculate_roots(equation):
    coeffs, rhs_value = parse_equation(equation)
    roots = np.roots(coeffs + [-rhs_value])
    roots_str = ', '.join([str(root) for root in roots])
    return roots_str

def predict_image(image, type):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    resized_image = cv2.resize(bw_image, (128, 128))
    img_data = scaler.transform(resized_image.reshape(1, -1))
    img_data = img_data.reshape(1, 128, 128, 1)

    if type == 'trigonometry':
        predictions = model_trig.predict(img_data)
        predicted_label = label_encoder_trig.inverse_transform([np.argmax(predictions)])
    else:
        predictions = model.predict(img_data)
        predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])
    
    return predicted_label[0]

def process_image(image, type):
    deep_copy = image.copy()

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(deep_copy, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    max_size = 250
    contour_predict = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 25 or h >= 25:
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            mask = cv2.bitwise_not(mask)
            contour_region = cv2.bitwise_or(image, mask)
            scale_factor = max(w, h) / max_size
            new_w = int(w / scale_factor)
            new_h = int(h / scale_factor)
            contour_region = cv2.resize(contour_region[y:y + h, x:x + w], (new_w, new_h))

            canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255
            x_offset = (300 - new_w) // 2
            y_offset = (300 - new_h) // 2
            cropped_region = cv2.resize(contour_region, (new_w, new_h))
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = cropped_region

            predicted_label = predict_image(canvas, type)

            label_position = (x, y - 10)  
            cv2.putText(deep_copy, str(predicted_label), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            contour_predict.append((((x + x + x + x + w + w) / 4, (-y - y - y - y - h - h) / 4), predicted_label))

    contour_predict = combine_adjacent_elements(contour_predict)
    contour_predict = combine_power(contour_predict)
    input_str = ""
    exp = replace_words_and_operators(contour_predict, input_str)
    expr = format_eq(exp)
    exp = replace_root(exp)
    result = calculate_expression(exp)

    if type == 'polynomial':
        parts = list(exp)
        for i in range(len(parts)):
            if parts[i] == '^':
                parts[i-1] = 'x'

        i = 0
        while i < len(parts) - 1:
            if parts[i] == '-' and parts[i+1] == '-':
                parts[i:i + 2] = ['=']
                break
            i += 1

        for i in range(len(parts)):
            if i >= 2 and parts[i] in ['+', '-', '='] and parts[i-2] != '^' and parts[i-1] != '=':
                parts[i-1] = 'x'

        modified_expr = ''.join(parts)
        expr = format_eq(modified_expr)
        roots = calculate_roots(modified_expr)

        return roots, expr
    
    elif type == 'trigonometry':
        parts = list(exp)

        if parts[2] == 'n':
            if parts[0] == 't':
                parts[1]='a'
            else:
                parts[0]='s'
                parts[1]='i'

        elif parts[0] == 'c':
            parts[1]='o'
            if parts[2] != 't':
                parts[2]='s'
                if len(parts)>3 and parts[3] == 'e':
                    parts[4]='c'
        
        elif parts[1] == 'e':
            parts[0]='s'
            parts[2]='c'

        elif parts[0] == 't':
            parts[1]='a'
            parts[2]='n'

        expr = ''.join(parts)
        print(expr)
        try:
            result = eval(expr, {"math": math, "sin": math.sin, "cos": math.cos, "tan": math.tan, "cot": lambda x: 1 / math.tan(x), "sec": lambda x: 1 / math.cos(x), "cosec": lambda x: 1 / math.sin(x)})
            result = str(result)
            return result, expr
        except Exception as e:
            ("Evaluation Error:", e)
            return expr, expr

    else:
        return result, expr
