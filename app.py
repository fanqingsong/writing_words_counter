
# !pip install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple #由于PaddleHub升级比较快，建议大家直接升级到最新版本的PaddleHub，无需指定版本升级
# !pip install shapely -i https://pypi.tuna.tsinghua.edu.cn/simple #该Module依赖于第三方库shapely，使用该Module之前，请先安装shapely
# !pip install pyclipper -i https://pypi.tuna.tsinghua.edu.cn/simple #该Module依赖于第三方库pyclipper，使用该Module之前，请先安装pyclipper
#
import os
import time

import paddlehub as hub
import cv2 as cv
import shutil


class WordsCounter:
    def __init__(self):
        self._ocr = hub.Module(name="chinese_ocr_db_crnn_server")
        self._workspace_path = "./workspace"
        self._snapshot_path = f"{self._workspace_path}/snapshot"

    def _get_image_data(self, image_path):
        image_data = cv.imread(image_path)

        print(type(image_data))
        print(f"image_data.shape={image_data.shape}")

        return image_data

    def _get_ocr_results_from_image_data(self, image_data):
        if image_data is None:
            print("image_data is none")
            return []

        ocr_results = self._ocr.recognize_text(images=[image_data])
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
        image_data = self._get_image_data(image_path)

        ocr_results = self._get_ocr_results_from_image_data(image_data)

        text = self._get_text_from_ocr_results(ocr_results)

        num = self._count_words_in_text(text)

        print(f"num = {num}")

        return num

    def _prepare_for_watch(self):
        if not os.path.exists(self._snapshot_path):
            os.mkdir(self._snapshot_path)

        #shutil.rmtree(self._snapshot_path)

    def watch_camera(self):
        self._prepare_for_watch()

        cap = cv.VideoCapture(0)

        fps = cap.get(cv.CAP_PROP_FPS)  # 视频平均帧率
        print(f"fps = {fps}")

        index = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                print(f"capture failed with ret={ret} frame={frame}")
                break

            ocr_results = self._get_ocr_results_from_image_data(frame)

            text = self._get_text_from_ocr_results(ocr_results)

            num = self._count_words_in_text(text)

            cv.putText(frame, f"Words total = {num}", (50, 50),
                        cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        2)

            cv.imshow('Video', frame)

            image_path = f'{self._snapshot_path}/{index}.jpg'
            cv.imwrite(image_path, frame)

            index += 1

            # 键盘输入空格暂停，输入q退出
            key = cv.waitKey(1) & 0xff
            if key == ord(" "):
                cv.waitKey(0)
            if key == ord("q"):
                break

            time.sleep(1 / fps)  # 按原帧率播放

        cap.release()
        cv.destroyAllWindows()
        print('capture finish, get %d frame' % index)




if __name__ == "__main__":
    words_counter = WordsCounter()

    # realtime counting
    words_counter.watch_camera()

    # testing one picture
    one_writing_path = './workspace/one_student_writing.jpeg'
    # words_counter.count_words_for_one_image(one_writing_path)









