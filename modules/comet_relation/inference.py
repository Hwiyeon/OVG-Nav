import torch
torch.set_num_threads(1)
torch.manual_seed(100)
import numpy as np
from PIL import Image
import argparse
import cv2

import clip


from torchvision import transforms
# from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from modules.comet_relation.comet_utils import use_task_specific_params, trim_batch

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

class Comet:
    def __init__(self, args):
        self.device = f'cuda:{args.model_gpu}'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.COMET_model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.COMET_model)
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


class CLIP:
    def __init__(self, args):
        # self.device = f'cuda:{args.model_gpu}'
        self.device = 'cpu'
        # self.model = CLIPModel.from_pretrained(args.CLIP_model).to(self.device)
        # self.processor = CLIPProcessor.from_pretrained(args.CLIP_model)


        # task = "summarization"
        # use_task_specific_params(self.model, task)
        # self.batch_size = 1
        # self.decoder_start_token_id = None

        self.model, self.preprocess = clip.load(args.CLIP_model, device=self.device, jit=True)


    def numpy_to_PIL_list(self, images):
        images = [Image.fromarray(image.astype(np.uint8)) for image in images]
        return images


    # def get_text_image_sim(self, text, images):
    #     PIL_images = self.numpy_to_PIL_list(images)
    #     inputs = self.processor(text=text, images=PIL_images, return_tensors="pt", padding=True).to(self.device)
    #     outputs = self.model(**inputs)
    #     text_image_sim = outputs.logits_per_text.softmax(dim=-1).detach().cpu().numpy()
    #     return text_image_sim
    #
    # def get_image_feat(self, images):
    #     images = self.numpy_to_PIL_list(images)
    #     inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
    #     outputs = self.vis_model(**inputs)
    #     return outputs.detach().cpu().numpy()

    def get_sim_from_feats(self, text_feat, img_feat, normalize=False):
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
        img_feat = img_feat / img_feat.norm(dim=1, keepdim=True)

        if normalize:
            # cosine similarity between same type features
            similarity = img_feat @ text_feat.T
        else:
            # text - img similarity
            similarity = self.model.logit_scale.exp() * img_feat @ text_feat.T

        return similarity




    def get_text_image_sim(self, text, images, out_img_feat=False):
        text = clip.tokenize(text).to(self.device)
        images = self.numpy_to_PIL_list(images)
        images = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.no_grad():
            torch.set_num_threads(1)
            img_feat = self.model.encode_image(images)
            text_feat = self.model.encode_text(text)
            # logits_per_image, logits_per_text = self.model(images, text)
            similarity = self.get_sim_from_feats(text_feat, img_feat)
            # sims = similarity.softmax(dim=-1).detach().cpu().numpy()
            sims = similarity.detach().cpu().numpy()

        if out_img_feat:
            return sims, img_feat.detach().cpu().numpy()
        else:
            return sims

    def get_image_feat(self, images):
        images = self.numpy_to_PIL_list(images)
        images = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.no_grad():
            torch.set_num_threads(1)
            image_features = self.model.encode_image(images)
        return image_features.type(torch.float32)

    def get_text_feat(self, texts):
        text = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            torch.set_num_threads(1)
            text_features = self.model.encode_text(text)
        return text_features.type(torch.float32)

    def get_text_image_feat_sim(self, text_feat, img_feat):
        text_feat = text_feat.to(self.device)
        img_feat = img_feat.to(self.device)
        if len(text_feat.shape) == 1:
            text_feat = text_feat.unsqueeze(0)
        if len(img_feat.shape) == 1:
            img_feat = img_feat.unsqueeze(0)
        similarity = self.get_sim_from_feats(text_feat, img_feat)
        sims = similarity.detach().cpu().numpy()

        return sims


class CommonSenseModel:
    def __init__(self, args):
        self.args = args

        self.comet = Comet(args)
        self.comet.model.eval()

        self.clip = CLIP(args)
        self.clip.model.eval()


    def gen_pred_words(self, query_object_name, num_generate=5):
        new_query_object_name = ''
        if len(query_object_name)>2:
            for i, letter in enumerate(query_object_name):
                if i and letter.isupper():
                    new_query_object_name += ' '
                new_query_object_name += letter.lower()
        else:
            new_query_object_name = query_object_name

        head = "A {}".format(new_query_object_name).lower()
        rel = "AtLocation"
        query = "{} {} [GEN]".format(head, rel)
        results = self.comet.generate([query], num_generate=num_generate)
        return results[0]

    def text_image_score(self, text, img, feat=False, return_only_max=True):
        if feat:
            sims_room = self.clip.get_text_image_feat_sim(text, img)
        else:
            sims_room = self.clip.get_text_image_sim(text, img)
        if return_only_max:
            max_score = np.round(np.max(sims_room, axis=1),5)
            argmax_score = np.argmax(sims_room, axis=1)
            # matched_text_room = [text[i] for i in np.argmax(sims_room, axis=1)]
            return max_score, argmax_score
        else:
            return np.round(sims_room,5), None


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    landmark_names = ['chair',          # 0
                      'sofa',          # 1
                      'plant',         # 2
                      'bed',           # 3
                      'toilet',        # 4
                      'tv monitor'     # 5
                      ]

    parser = argparse.ArgumentParser()

    ## eval configs ##
    parser.add_argument("--model_gpu", type=str, default="0")
    parser.add_argument("--sim_gpu", type=str, default="0")

    ## model configs ##
    parser.add_argument("--free_space_model", type=str,
                        default="models/free_space_model/ckpts/split_lr0.001_0227_range_2.0/best_model_1.pth")
    parser.add_argument("--CLIP_model", type=str, default="ViT-B/32")
    parser.add_argument("--COMET_model", type=str, default="comet-atomic_2020_BART")

    args = parser.parse_args()

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




    co = CommonSenseModel(args)

    query_object_name = 'sofa'
    pred_words = co.gen_pred_words(query_object_name)
    print('Query Object: ', query_object_name)

    pano_img = cv2.imread('test_data/pano_rgb.png')
    pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)
    dirc_imgs = get_dirc_imgs_from_pano(pano_img)

    dirc_score, dirc_text_arg_score = co.text_image_score(pred_words, dirc_imgs)
    matched_text = [pred_words[i] for i in dirc_text_arg_score]
    pano_score, pano_text_arg_score = co.text_image_score(pred_words, [pano_img])
    pano_matched_text = [pred_words[i] for i in pano_text_arg_score]

    img_feat = co.clip.get_image_feat(dirc_imgs)
    text_feat = co.clip.get_text_feat(pred_words).type(torch.float32)
    mean_img_feat = torch.mean(img_feat, dim=0).type(torch.float32).unsqueeze(0)
    mean_score, mean_text_arg_score = co.text_image_score(text_feat, mean_img_feat, feat=True)
    mean_matched_text = [pred_words[i] for i in mean_text_arg_score]

    print('AtLocation  : ', pred_words)
    print(matched_text)
    print(dirc_score)


