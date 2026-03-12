import json
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# from itergen.itergen.syncode import syncode
from itergen.syncode import syncode
from itergen.syncode.syncode.infer import Syncode
from itergen.main import IterGen
from privacy.sys_prompts import PROMPT_TEMPLATE, PII_INFO, PII_PROMPT_TYPE, PRIVACY_TOPIC, JAILBREAK_PROMPTS

from privacy.utils import get_privacy_understanding_messages, load_enron_email_prompts, \
    get_pii_messages_k_shot
from privacy.result_agg import calculate_scores
from typing import Union

import time
from tqdm import tqdm
os.environ['HF_CACHE'] = '/data/share/models/hugging_face/'

class MessageConstructor(object):
    def __init__(self, sys_demo, task_desc,is_raw_input=False):
        self.sys_demo = sys_demo
        self.task_desc = task_desc
        self.is_raw_input=is_raw_input
    def get_message(self, input):
        messages = []
        if len(self.sys_demo) > 0:
            messages.append({"role": "system", "content": self.sys_demo})

        if len(self.task_desc) > 0:
            messages.append({"role": "user", "content": self.task_desc})
        if self.is_raw_input:
            messages.append({"role": "user", "content": input})   
        else:    
            messages = messages + input
            
        return messages

ScenarioName2ResultDir = {
    "pii": "./results/privacy/generations/pii",
    "pii_jailbreak": "./results/privacy/generations/pii_jailbreak",
    "privacy_understanding": "./results/privacy/generations/understanding",
    "enron_email_extraction": "./results/privacy/",
}


def format_prompt(prompt: str, model: str) -> Union[str, list]:
    """
    If the model is instruct-tuned model then we format the prompt as follows:
    [{role: user, content: prompt}]

    else:
    prompt = prompt
    """
    if 'instruct' in model or 'Instruct' in model or 'it' in model or 'chat' in model:
        return [{"role": "user", "content": prompt}]
    return prompt


def generate_with_itergen(iter_gen, prompt, email_list):
    # print(args.allow_chat_template)
    if (args.allow_chat_template):
        iter_gen.start(format_prompt(prompt=prompt, model=args.model_id))
    else:
        iter_gen.start(prompt)
    num_backwards = 0
    backwards_limit = 10

    while not iter_gen.finished():
        out = iter_gen.forward(unit='EMAIL')

        email_addresses = iter_gen.view('EMAIL')[0]
        last_email_id = email_addresses[-1] if email_addresses else None        
        if last_email_id!=None and num_backwards < backwards_limit and last_email_id in email_list:
            out = iter_gen.backward('EMAIL')
            num_backwards += 1
            continue  
        # print(out)
    # print(iter_gen._metadata)
    return out, iter_gen._metadata


def generate_and_save(args, dataset, gt_list, constructor, out_file, batch_size=1, keynames=['gt'], ner=False):
    dataset_size= len(dataset)
    
    # Batch size and dataset size
    batch_size = 1
    num_batches = int(dataset_size / batch_size)
    num_batches = args.num_problems

    price_cost = 0
    prompt_tokens = 0
    comple_tokens = 0
    num_prompt = 0
    total_tokens = 0
    start_time = time.time()
    if ner==True: 
        try:
            from transformers import pipeline
            ner_classifier = pipeline("token-classification", model = "bigcode/starpii", aggregation_strategy="simple")
        except Exception as err:
            print(err)

    if args.mode == 'original':
        syncode =  Syncode(mode='original', model=args.model_id, parse_output_only=True, do_sample=False, max_new_tokens=args.privacy.max_tokens, device=args.device)
    else:
        # Lark grammar for words seperated by space
        grammar = """
            start: (OTHER | EMAIL)*
            OTHER:  /[^ ]/ 
            EMAIL: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+(\.[a-zA-Z0-9.-]+)+/
            %import common.WS
            %ignore WS
        """
        model = IterGen(grammar=grammar, model_id=args.model_id, parse_output_only=True, do_sample=False, max_new_tokens=args.privacy.max_tokens, recurrence_penalty=args.recurrence_penalty, device=args.device)
        email_list = gt_list[0]
    
    with open(out_file, "w") as f: 
        for batch_idx in tqdm(range(num_batches)):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_data = dataset[batch_start:batch_end]

            if args.mode == 'original':
                output = syncode.infer(batch_data[0])
                total_tokens += syncode.model.total_tokens
                
            else:
                if ('idrc' in batch_data[0]):
                    x = 1
                output, num_tok = generate_with_itergen(model, batch_data[0], email_list)
                total_tokens += num_tok['total_tokens']


            cost = None
            # cost, cache = model.do_generation(batch_data, constructor, n=args.privacy.n, t=args.privacy.t,
            #                                     max_tokens=args.privacy.max_tokens,
            #                                     dry_run=args.dry_run)  # only works for n=1 now...

            # computing resource
            # price_cost += cost[0]
            # prompt_tokens += cost[1]
            # comple_tokens += cost[2]
            num_prompt += 1

            res={}
            for i, key in enumerate(keynames):
                
                batch_gt = gt_list[i][batch_start:batch_end] 
                res[key] = batch_gt[0]

            # res_gen = { "continuation": x[1], "isbanned": x[2], "prompt": x[3], "response": x[-1]}
            res_gen = {"prompt": batch_data[0], "response": output[0]}

            res.update(res_gen)
            if ner==True: 
                try:
                    ner_results = ner_classifier(output[0])
                    ner_words=[] 
                    if len(ner_results)>0:
                        ner_words= [entity['word'] for entity in ner_results]
                    res.update({"named_entity_rec_continuation":  ner_words })
                except Exception as err:
                    print(err)
            f.write(json.dumps(res) + "\n")
    end_time = time.time()
    return total_tokens, end_time-start_time, num_prompt

def main(args):
    print(args)
    out_file = args.privacy.out_file
    dataset_size = args.privacy.dataset_size

    all_cost = {}
    all_prompt_tokens = {}
    all_comple_tokens = {}
    all_num_prompt = {}

    if args.privacy.scenario_name == "pii":

        for pii in args.privacy.personal_infos:
            if pii not in PII_INFO:
                print(f"Skip {pii} as it is not defined in supported PII types {PII_INFO}")
                continue
            constructor = MessageConstructor(PROMPT_TEMPLATE[args.privacy.template]['sys_prompt'].format(pii),
                                             PROMPT_TEMPLATE[args.privacy.template]['task_desc'])

            for fewshot_type in args.privacy.prompt_types:
                if fewshot_type not in PII_PROMPT_TYPE:
                    print(f"Skip {fewshot_type} as it is not defined in supported prompt types {PII_PROMPT_TYPE}")
                    continue

                dataset, gt_list = get_pii_messages_k_shot(filename=args.privacy.data_file, k=args.privacy.few_shot_num, pp=pii,
                                                           fewshot_type=fewshot_type, num_cases=dataset_size)

                result_dir = ScenarioName2ResultDir[args.privacy.scenario_name]
                os.makedirs(result_dir, exist_ok=True)
                out_file = os.path.join(
                    result_dir,
                    "{}_{}shot{}_s{}_{}_{}.jsonl".format(
                        pii, args.privacy.few_shot_num, fewshot_type, dataset_size, args.model_config.model, args.privacy.seed
                    ).replace("/", "_").replace(" ", "")
                )
                print(f"Saving to {out_file}")

                price_cost, prompt_tokens, comple_tokens, num_prompt=  generate_and_save(args=args, 
                                                                                         dataset= dataset, 
                                                                                         gt_list=[gt_list],
                                                                                         constructor=constructor, 
                                                                                         out_file=out_file, 
                                                                                         batch_size= args.privacy.batch_size,
                                                                                         keynames=['gt']
                                                                                         )

                all_cost[pii] = price_cost
                all_prompt_tokens[pii] = prompt_tokens
                all_comple_tokens[pii] = comple_tokens
                all_num_prompt[pii] = num_prompt
    elif args.privacy.scenario_name == "pii_jailbreak":

        for pii in args.privacy.personal_infos:
            if pii not in PII_INFO:
                print(f"Skip {pii} as it is not defined in supported PII types {PII_INFO}")
                continue
            for jb_idx, jailbreak_prompt in enumerate(JAILBREAK_PROMPTS):
                constructor = MessageConstructor(jailbreak_prompt, PROMPT_TEMPLATE[args.privacy.template]['task_desc'])

                for fewshot_type in args.privacy.prompt_types:
                    if fewshot_type not in PII_PROMPT_TYPE:
                        print(f"Skip {fewshot_type} as it is not defined in supported prompt types {PII_PROMPT_TYPE}")
                        continue

                    dataset, gt_list = get_pii_messages_k_shot(filename=args.privacy.data_file, k=args.privacy.few_shot_num, pp=pii,
                                                            fewshot_type=fewshot_type, num_cases=dataset_size)

                    result_dir = ScenarioName2ResultDir[args.privacy.scenario_name]
                    os.makedirs(result_dir, exist_ok=True)
                    base_fname = "{}_{}shot{}_s{}_{}_{}_jb{}.jsonl".format(
                            pii, args.privacy.few_shot_num, fewshot_type, dataset_size, args.model_config.model, args.privacy.seed, jb_idx
                        ).replace("/", "_").replace(" ", "")
                    out_file = os.path.join(result_dir, base_fname)
                    
                    print(f"Saving to {out_file}")

                    price_cost, prompt_tokens, comple_tokens, num_prompt=  generate_and_save(args=args, 
                                                                                            dataset= dataset, 
                                                                                            gt_list=[gt_list],
                                                                                            constructor=constructor, 
                                                                                            out_file=out_file, 
                                                                                            batch_size= args.privacy.batch_size,
                                                                                            keynames=['gt'],
                                                                                            ner=True
                                                                                            )
                    
                    all_cost[pii] = price_cost
                    all_prompt_tokens[pii] = prompt_tokens
                    all_comple_tokens[pii] = comple_tokens
                    all_num_prompt[pii] = num_prompt


    elif args.privacy.scenario_name == "privacy_understanding":
        constructor = MessageConstructor(PROMPT_TEMPLATE[args.privacy.template]['sys_prompt'],
                                         PROMPT_TEMPLATE[args.privacy.template]['task_desc'])
        for topic in args.privacy.privacy_topics:
            if topic not in PRIVACY_TOPIC:
                print(f"Skip {topic} as it is not defined in supported topics {PRIVACY_TOPIC}")
                continue
            dataset, gt_list = get_privacy_understanding_messages(topic, args.privacy.question_prompt)

            result_dir = ScenarioName2ResultDir[args.privacy.scenario_name]
            os.makedirs(result_dir, exist_ok=True)
            out_file = os.path.join(
                result_dir,
                "{}_t{}_s{}_{}_ques{}_{}.jsonl".format(
                    topic, args.privacy.template, dataset_size, args.model_config.model, args.privacy.question_prompt, args.privacy.seed
                ).replace("/", "_").replace(" ", "")
            )

            expanded_dataset=[]
            expanded_gt_list=[]
            for i, data in enumerate(dataset):
                expanded_dataset.extend([data] * dataset_size)
                expanded_gt_list.extend([gt_list[i]] * dataset_size)
    
            print(f"Saving to {out_file}")
            price_cost, prompt_tokens, comple_tokens, num_prompt=  generate_and_save(args=args, 
                                                                                         dataset= expanded_dataset, 
                                                                                         gt_list=[expanded_gt_list], 
                                                                                         constructor=constructor, 
                                                                                         out_file=out_file, 
                                                                                         batch_size= dataset_size,
                                                                                         keynames=['word']
                                                                                         )
        
            all_cost[topic] = price_cost
            all_prompt_tokens[topic] = prompt_tokens
            all_comple_tokens[topic] = comple_tokens
            all_num_prompt[topic] = num_prompt

    elif args.privacy.scenario_name == "enron_email_extraction":
        constructor = MessageConstructor(PROMPT_TEMPLATE[args.privacy.template]['sys_prompt'],
                                              PROMPT_TEMPLATE[args.privacy.template]['task_desc'], is_raw_input=True)
        for prompt_type in args.privacy.prompt_types:
            print("prompt_type", prompt_type)
            dataset, name_list, email_list = load_enron_email_prompts(args.privacy.data_file, prompt_type)
            if dataset_size < 1 or dataset_size > len(dataset):
                dataset_size = len(dataset)
            else:
                dataset = dataset[:dataset_size]

            result_dir = ScenarioName2ResultDir[args.privacy.scenario_name]
            os.makedirs(result_dir, exist_ok=True)
            out_file = os.path.join(
                                        result_dir,
                                        "{}_s{}_{}_{}_{}_{}_completions.jsonl".format(
                                            args.config, 
                                            args.num_problems, 
                                            args.model_id, 
                                            args.privacy.seed, 
                                            args.mode,  # Include args.mode here
                                            args.allow_chat_template
                                        ).replace("/", "_").replace(" ", "")
                                    )

            print(f"Saving to {out_file}")
            total_tokens, time, num_prompt =  generate_and_save(args=args, 
                                                                                        dataset= dataset, 
                                                                                         gt_list=[ email_list,name_list],
                                                                                         constructor=constructor, 
                                                                                         out_file=out_file, 
                                                                                         batch_size= args.privacy.batch_size,
                                                                                         keynames=['gt-email', 'gt-name']
                                                                                         )


    # print("*********final results summary******")

    # cost_over_all_settings = 0
    # prompt_tokens_over_all_settings = 0
    # comple_tokens_over_all_settings = 0
    # num_prompt_over_all_settings = 0
    # for key in all_cost.keys():
    #     print("scenario: ", key)
    #     print("Cost: ", all_cost[key])
    #     print("Num of prompt tokens: ", all_prompt_tokens[key])
    #     print("Num of completion tokens: ", all_comple_tokens[key])
    #     cost_over_all_settings += all_cost[key]
    #     prompt_tokens_over_all_settings += all_prompt_tokens[key]
    #     comple_tokens_over_all_settings += all_comple_tokens[key]
    #     num_prompt_over_all_settings += all_num_prompt[key]

    # print("*********sum results summary******")
    # print("sum - Cost: ", cost_over_all_settings)
    # print("sum - Num of prompt tokens: ", prompt_tokens_over_all_settings)
    # print("sum - Num of completion tokens: ", comple_tokens_over_all_settings)
    # print("sum - Num of prompt: ", num_prompt_over_all_settings)
    
    print("*********calculate score ******")
    calculate_scores(out_file=out_file, total_tokens=total_tokens, time=time, num_prompt=num_prompt, args=args)


if __name__ == "__main__":
    # Convert yaml to argparse
    import argparse
    import yaml

    parser = argparse.ArgumentParser()

    cur_dir = os.path.dirname(os.path.realpath(__file__)) + "/configs/"
    # parser.add_argument("--config", type=str, default=f"email_extraction_context50")
    parser.add_argument("--config", type=str, default=f"email_extraction_known_fiveshot_a")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--privacy", type=type(parser), default=None)
    parser.add_argument("--mode", type=str, default="itergen")
    parser.add_argument("--recurrence_penalty", type=float, default=0.3)
    parser.add_argument("--allow_chat_template", action='store_true', help="Allow chat template")
    parser.add_argument("--no_allow_chat_template", action='store_false', dest='allow_chat_template', help="Disallow chat template")
    parser.set_defaults(allow_chat_template=False)  # Default is True


    args = parser.parse_args()

    config_path = f"{cur_dir}{args.config}.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert dict to structured object
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    args.privacy = Struct(**config)

    # Run the main function
    main(args)
