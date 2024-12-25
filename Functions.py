import cv2
import random
import numpy as np

foreground_extractor = cv2.createBackgroundSubtractorKNN(history=100000, detectShadows=False, dist2Threshold=450.0)
id = 0
nb_frame_enter_tow_color = 1
n = 0


def detection_visage(frame, face_cascade, img, alpha):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for face in faces:
        _width = face[1] + face[3]
        _height = face[0] + face[2]
        bg = frame[face[1]:_width, face[0]:_height]
        height = bg.shape[0]
        width = bg.shape[1]
        height = abs(height)
        width = abs(width)
        image = cv2.resize(img, (width, height))
        alpha = cv2.resize(alpha, (width, height))
        bg = cv2.multiply(1.0 - alpha, bg.astype(float))
        out_image = cv2.add(image, bg)
        frame[face[1]:_width, face[0]:_height] = out_image

    return frame


def color_filter(frame):
    global id, n, nb_frame_enter_tow_color
    if id == 0:
        blue_channel, _, _ = cv2.split(frame)
        frame = cv2.merge([blue_channel, np.zeros_like(blue_channel), np.zeros_like(blue_channel)])
        if n == nb_frame_enter_tow_color:
            id = 1
            n = 0
        else:
            n += 1
    elif id == 1:
        _, _, red_channel = cv2.split(frame)
        frame = cv2.merge([np.zeros_like(red_channel), np.zeros_like(red_channel), red_channel])
        if n == nb_frame_enter_tow_color:
            id = 0
            n = 0
        else:
            n += 1
    return frame


def filter_jail(frame, img):
    img = cv2.resize(img, (frame.shape[1], frame.shape[0]))
    alpha = img[:, :, 2] / 255.0
    for c in range(0, 3):
        frame[:, :, c] = (1 - alpha) * frame[:, :, c] + alpha * img[:, :, c]
    return frame


def filter_background(frame, money_filter, x, y, speed):
    mask = foreground_extractor.apply(frame)
    _, foreground_mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60)))
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80)))
    foreground = cv2.bitwise_and(frame, frame, mask=foreground_mask)
    foreground_mask = cv2.medianBlur(foreground_mask, 5)
    new_background = cv2.imread('Images/Projet/coffre-fort-banque.jpg')
    new_background = cv2.resize(new_background, (frame.shape[1], frame.shape[0]))

    if money_filter: new_background = filter_money(new_background, speed, x, y)

    background_mask = cv2.bitwise_not(foreground_mask)
    background = cv2.bitwise_and(new_background, new_background, mask=background_mask)

    result = cv2.add(foreground, background)
    return result


def filter_money(frame, speed, x_positions, y_positions):
    money_image = cv2.imread('Images/Projet/billet.png')
    money_image = cv2.resize(money_image, (50, 50))
    alpha_money = money_image[:, :, 2] / 255.0

    for i in range(len(x_positions)):
        n = random.randint(0, 25)
        random_x = x_positions[i]
        money_height, money_width, _ = money_image.shape
        start_y = y_positions[i] if y_positions[i] >= 0 else 0
        end_y = y_positions[i] + money_height if y_positions[i] + money_height <= frame.shape[0] else frame.shape[0]
        money_visible_height = end_y - start_y
        if y_positions[i] < 0:
            random_x = random.randint(0, frame.shape[0])
            x_positions[i] = random_x
            y_positions[i] = 1
        bg = frame[start_y:end_y, random_x:random_x + money_width]
        bg_width, bg_height, _ = bg.shape;
        fg = money_image[:money_visible_height, :]
        for c in range(0, 3):
            w = abs(bg[:, :, c].shape[1])
            h = abs(bg[:, :, c].shape[0])
            if h != 0:
                alpha_money = cv2.resize(alpha_money, (w, h))
                bg[:, :, c] = (1 - alpha_money) * bg[:, :, c] + alpha_money * fg[:, :, c]
        frame[start_y:end_y, random_x:random_x + money_width] = bg
        if n == 1 or y_positions[i] > 0:
            y_positions[i] = y_positions[i] + speed
            if y_positions[i] > frame.shape[0]:
                y_positions[i] = -money_height
    return frame
