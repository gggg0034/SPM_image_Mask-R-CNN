
import fast_json_to_dataset
import Images_Augmentation.color_aug_img_json as color_aug_img_json
import Images_Augmentation.geo_aug_img_json as geo_aug_img_json
import pre_draw_mask
import re_write_mask
import train_test

if __name__ == '__main__':
    geo_aug_img_json.main()                 #L1增强
    color_aug_img_json.main()               #L2、L3增强
    pre_draw_mask.main()                    #预处理dataset
    re_write_mask.main()                    #
    fast_json_to_dataset.main()             #
    train_test.train_main()                 #开始训练
    