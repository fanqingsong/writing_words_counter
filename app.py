
# !pip install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple #由于PaddleHub升级比较快，建议大家直接升级到最新版本的PaddleHub，无需指定版本升级
# !pip install shapely -i https://pypi.tuna.tsinghua.edu.cn/simple #该Module依赖于第三方库shapely，使用该Module之前，请先安装shapely
# !pip install pyclipper -i https://pypi.tuna.tsinghua.edu.cn/simple #该Module依赖于第三方库pyclipper，使用该Module之前，请先安装pyclipper
#


import paddlehub as hub
import cv2


class WordsCounter:
    def __init__(self):
        self._ocr = hub.Module(name="chinese_ocr_db_crnn_server")
        self._workspace_path = "./workspace"
        self._snapshot_path = f"{self._workspace_path}/snapshot"

    def _get_ocr_results_from_image(self, image_path):
        img_data = cv2.imread(image_path)

        ocr_results = self._ocr.recognize_text(images=[img_data])
        print(ocr_results)

        return ocr_results

    def _get_text_from_ocr_results(self, ocr_results):
        all_text = []

        for one_result in ocr_results:
            data = one_result["data"]

            for one_info in data:
                one_text = one_info["text"]
                all_text.append(one_text)

            # add empty line before storing next image text
            all_text.append("")

        all_text = "\r\n".join(all_text)

        print("----- all text --------")
        print(all_text)

        return all_text

    def _count_words_in_text(self, text: str):
        pure_text = text.replace("\r\n", "")

        return len(pure_text)

    def count_words_for_one_image(self, image_path):
        ocr_results = self._get_ocr_results_from_image(image_path)

        text = self._get_text_from_ocr_results(ocr_results)

        num = self._count_words_in_text(text)

        print(f"num = {num}")

        return num

    def watch_camera(self):
        cap = cv2.VideoCapture(0)

        index = 0
        while True:
            ret, frame = cap.read()
            print(f"capture ret={ret} frame={frame}")

            if ret:
                cv2.imwrite(f'{self._snapshot_path}/{index}.jpg', frame)
                print(type(frame))
                print(frame.shape)
                index += 1
            else:
                break

        cap.release()
        print('video split finish, all %d frame' % index)




if __name__ == "__main__":
    # for test
    one_writing_path = './workspace/one_student_writing.jpeg'

    words_counter = WordsCounter()
    #words_counter.watch_camera()

    #
    words_counter.count_words_for_one_image(one_writing_path)









