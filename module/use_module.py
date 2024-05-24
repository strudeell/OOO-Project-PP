from module import preprocess_image, extract_and_filter_boxes, recognize_characters, draw_boxes_on_image

image_path = input("Укажите путь до изображения: ")

original_img, img = preprocess_image(image_path)
boxes, y1, y2 = extract_and_filter_boxes(img)
result = recognize_characters(original_img, boxes, y1, y2, model_path, class_names)
print(result)
draw_boxes_on_image(original_img, boxes)