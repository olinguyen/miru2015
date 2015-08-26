import os.path as path
import numpy as np
import os
import glob
import re


def get_video_list(stream="spatial"):
    #files = glob.glob("/home/syq/data/ucf101-frames/*/g2[0-1]/*/*.jpg")
    with open('/home/syq/trecvid/yehao-two-stream/examples/temporal/source.txt') as f:
        files = f.read().splitlines()

    if stream == "temporal":
        matches = []
        for stacked_list in files:
            match = re.search('g0[1-7]', stacked_list)
            if match:
                matches.append(match.string)
        with open('testlist01_temporal.txt', 'w') as f:
            for line in matches:
                f.write(line + "\n")


def get_groundtruth(stream="spatial"):
    class_dict = {}
    class_index_filepath = "/home/syq/fudan/ucfTrainTestlist/classInd.txt"
    with open(class_index_filepath) as f:
        lines = f.readlines()

    for line in lines:
        (class_index, class_name) = line.split()
        class_dict[class_name] = int(class_index) - 1

    for testlist in range(1, 2):
        if stream == "spatial":
            with open("/home/syq/data/testlist0%d_imgs.txt" % testlist) as f:
                images = f.readlines()

            y_true = np.zeros(len(images))
            for index, img_path in enumerate(images):
                label = path.basename(path.abspath(path.join(img_path,
                                                                "../../../")))
                y_true[index] = class_dict[label]

        elif stream == "temporal":
            videos = glob.glob("/home/syq/data/optflow/*/*/*")
            videos.sort()
            y_true = np.zeros(len(videos))
            for index, video in enumerate(videos):
                label = path.basename(path.abspath(path.join(video, "../../")))
                y_true[index] = class_dict[label]

    return y_true


def get_predictions(stream="spatial"):
    num_classes = 101
    if stream == "spatial":
        num_tests = 693433
        prob_filepath = "/home/syq/trecvid/yehao-two-stream/examples/spatial/feature/ucf101_prob.binary"
    elif stream == "temporal":
        num_tests = 13320
        prob_filepath = "/home/syq/trecvid/yehao-two-stream/examples/temporal/feature/full_probabilities.binary"

    y_prob = np.fromfile(prob_filepath, np.float32) \
               .reshape(num_tests, num_classes)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred


def main():
    stream = "spatial"
    y_true = get_groundtruth(stream)
    y_pred = get_predictions(stream)
    print len(y_true)
    print len(y_pred)
    if (len(y_true) != len(y_pred)):
        print "y_true length %d does not match y_pred length %d" \
            % (len(y_true), len(y_pred))
        return False

    acc = (y_true == y_pred).mean()
    print acc


if __name__ == "__main__":
    main()
