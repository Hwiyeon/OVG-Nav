import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from PIL import Image
import spacy
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from comet_utils import use_task_specific_params, trim_batch


# use gpu 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path,device = 'cuda'):
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self,
            queries,
            decode_method="beam",
            num_generate=5,
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs


class co_occurance_score():
    def __init__(self,model_dir, device):
        self.nlp = spacy.load('en_core_web_md')
        print("model loading ...")
        self.comet = Comet(model_dir, device=device)
        self.comet.model.zero_grad()
        print("model loaded")


    def landmark_init(self,landmark_cat):
        self.landmark_cat = landmark_cat

    def score(self,query_object_name):
        new_query_object_name = ''
        if len(query_object_name)>2:
            for i, letter in enumerate(query_object_name):
                if i and letter.isupper():
                    new_query_object_name += ' '
                new_query_object_name += letter.lower()
        else:
            new_query_object_name = query_object_name

        head = "A {}".format(new_query_object_name).lower()
        rel = ["AtLocation","LocatedNear"]
        query_1 = "{} {} [GEN]".format(head, rel[1])
        # query_2 = "{} {} [GEN]".format(head, rel[1])
        queries = [query_1]#, query_2]
        results = self.comet.generate(queries, decode_method="beam", num_generate=20)
        # print(results)
        res = []
        for l in self.landmark_cat:
            sims = []
            for r in results[0]:
                doc1 = self.nlp(r)
                doc2 = self.nlp(l)
                sims.append(doc1.similarity(doc2))
            # for r in results[1]:
            #     doc1 = self.nlp(r)
            #     doc2 = self.nlp(l)
                # sims.append(doc1.similarity(doc2))
            res.append(round(max(sims),3))
        return res


class co_occurance_feature():
    def __init__(self, model_dir, model_type='clip', device='cuda:0'):
        print("model loading ...")
        self.model_dir = model_dir
        comet_model_dir = os.path.join(model_dir, 'comet-atomic_2020_BART')
        self.comet = Comet(comet_model_dir, device=device)
        self.comet.model.zero_grad()
        print("model loaded")

        self.model_type = model_type

        if model_type == 'clip':
            self.init_clip_model()
        elif model_type == 'lang':
            self.init_nlp_model()


    def init_nlp_model(self):
        self.nlp = spacy.load('en_core_web_md')
        print("nlp model loaded")

    def init_clip_model(self):
        clip_model_dir = os.path.join(self.model_dir, 'clip-vit-base-patch32')
        self.clip = CLIP(clip_model_dir)
        print("clip model loaded")


    def gen_pred_words(self, query_object_name):
        new_query_object_name = ''
        if len(query_object_name)>2:
            for i, letter in enumerate(query_object_name):
                if i and letter.isupper():
                    new_query_object_name += ' '
                new_query_object_name += letter.lower()
        else:
            new_query_object_name = query_object_name

        head = "A {}".format(new_query_object_name).lower()
        rel = ["AtLocation","LocatedNear"]
        query_room = "{} {} [GEN]".format(head, rel[0])
        query_objs = "{} {} [GEN]".format(head, rel[1])
        queries = [query_room, query_objs]
        results = self.comet.generate(queries, decode_method="beam", num_generate=20)
        return results


    def score(self, query_object_name, target, target_type='lang'):
        results = self.gen_pred_words(query_object_name)

        if target_type == 'lang':
            res_room, res_objs = [], []
            matched_text_room, matched_text_objs = [], []
            for t in target:
                sims_room = []
                sims_objs = []
                for r in results[0]:
                    doc1 = self.nlp(r)
                    tf = self.nlp(t)
                    sims_room.append(doc1.similarity(tf))
                for r in results[1]:
                    doc1 = self.nlp(r)
                    tf = self.nlp(t)
                    sims_objs.append(doc1.similarity(tf))
                res_room.append(round(max(sims_room),3))
                res_objs.append(round(max(sims_objs),3))
                matched_text_room.append(results[0][np.argmax(sims_room)])
                matched_text_objs.append(results[1][np.argmax(sims_objs)])

        elif target_type == 'image':
            sims_room = self.clip.get_text_image_sim(results[0], target)
            sims_objs = self.clip.get_text_image_sim(results[1], target)

            res_room = np.round(np.max(sims_room, axis=0),3)
            res_objs = np.round(np.max(sims_objs, axis=0),3)
            matched_text_room = [results[0][i] for i in np.argmax(sims_room, axis=0)]
            matched_text_objs = [results[1][i] for i in np.argmax(sims_objs, axis=0)]

        else:
            raise ValueError("target_type must be either 'lang' or 'image'")
        return [res_room, res_objs], [matched_text_room, matched_text_objs]




class CLIP:
    def __init__(self, model_dir, device='cuda'):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_dir)
        self.processor = CLIPProcessor.from_pretrained(model_dir)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def numpy_to_PIL_list(self, images):
        images = [Image.fromarray(image) for image in images]
        return images

    def get_text_image_sim(self, text, images):
        images = self.numpy_to_PIL_list(images)
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        text_image_sim = outputs.logits_per_text.softmax(dim=-1).detach().cpu().numpy()
        return text_image_sim

if __name__ == '__main__':
    landmark_names = ['chair',          # 0
                      'sofa',          # 1
                      'plant',         # 2
                      'bed',           # 3
                      'toilet',        # 4
                      'tv monitor'     # 5
                      ]

    obj_names = ['chair',         # 0
                     'couch',         # 1
                     'potted plant',  # 2
                     'bed',           # 3
                     'dining table',  # 4
                     'toilet',        # 5
                     'tv',            # 6
                     'laptop',        # 7
                     'microwave',     # 8
                     'oven',          # 9
                     'sink',          # 10
                     'refrigerator',  # 11
                     'clock',         # 12
                     'vase',          # 13
                     ]

    room_names = ['livingroom',  # 0
                  'bedroom',  # 1
                  'kitchen',  # 2
                  'diningroom',  # 3
                  'bathroom',  # 4
                  'office',  # 5
                  'hallway',  # 6
                  'others'  # 7
                  ]


    def get_images_from_video(video_path):
        cap = cv2.VideoCapture(video_path)
        images = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return images


    def get_dirc_imgs_from_pano(pano_img, num_imgs=12):
        pw, ph = pano_img.shape[1], pano_img.shape[0]

        # split the panorama into 12 square images with even angles
        dirc_imgs = []
        for i in range(num_imgs):
            angle = i * 360 / num_imgs
            x = int(pw * (angle / 360))
            dirc_img = pano_img[:, x:x + ph]
            if x + ph > pw:
                dirc_img = np.concatenate((dirc_img, pano_img[:, :x + ph - pw]), axis=1)
            dirc_imgs.append(dirc_img)
        #         print(np.shape(dirc_img))
        return np.array(dirc_imgs)




    co = co_occurance_feature("./", model_type='lang', device="cuda:0")


    query_object_name = 'sofa'
    pred_words = co.gen_pred_words(query_object_name)
    print('Query Object: ', query_object_name)


    if co.model_type == 'clip':
        pano_video_path = './test_data/2azQ1b91cZZ_0000/pano_rgb.avi'
        pano_images = get_images_from_video(pano_video_path)
        dirc_imgs = get_dirc_imgs_from_pano(pano_images[1])

        result_objs, matched_text = co.score(query_object_name, dirc_imgs, target_type='image')
        print('AtLocation  : ', pred_words[0])
        print(matched_text[0])
        print(result_objs[0])
        print('LocatedNear : ', pred_words[1])
        print(matched_text[1])
        print(result_objs[1])

    elif co.model_type == 'lang':
        result_objs, matched_text_objs = co.score(query_object_name, obj_names, target_type='lang')
        print(obj_names)
        print('AtLocation  : ', pred_words[0])
        print(matched_text_objs[0])
        print(result_objs[0])
        print('LocatedNear : ', pred_words[1])
        print(matched_text_objs[1])
        print(result_objs[1])

        result_rooms, matched_text_rooms = co.score(query_object_name, room_names, target_type='lang')
        print(room_names)
        print('AtLocation  : ', pred_words[0])
        print(matched_text_rooms[0])
        print(result_rooms[0])
        print('LocatedNear : ', pred_words[1])
        print(matched_text_rooms[1])
        print(result_rooms[1])




