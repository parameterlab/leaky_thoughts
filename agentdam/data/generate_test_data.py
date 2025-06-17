# Copyright (c) Meta Platforms, Inc. and affiliates.
import os, json, argparse


MAP_SITES = {"__REDDIT__": os.environ.get("REDDIT", ""),
             "__SHOPPING__": os.environ.get("SHOPPING", ""),
             "__SHOPPING_ADMIN__": os.environ.get("SHOPPING_ADMIN", ""),
             "__GITLAB__": os.environ.get("GITLAB", ""),
             "__WIKIPEDIA__": os.environ.get("WIKIPEDIA", ""),
             "__MAP__": os.environ.get("MAP", ""),
             "__HOMEPAGE__": os.environ.get("HOMEPAGE", ""),
             "__CLASSIFIEDS__": os.environ.get("CLASSIFIEDS", "")}


def site_mapping(D):
    if isinstance(D, (list, dict)):
        items = enumerate(D) if isinstance(D, list) else D.items()
        for k, v in items:
            if isinstance(v, str):
                for site_name, url in MAP_SITES.items():
                    if site_name in v:
                        D[k] = v.replace(site_name, url)
            elif isinstance(v, (dict, list)):
                site_mapping(v)


def main(args):
    combined_dir = args.combined_dir

    for website in ["reddit", "shopping", "gitlab"]:
        privacy_test_prefix = website + "_privacy"
        print("Generating", privacy_test_prefix, "...")
        with open(os.path.join(combined_dir, privacy_test_prefix + ".json"), 'r') as json_file:
            vwa_tasks_config = json.load(json_file)

        indiv_configs_path = os.path.join(combined_dir, privacy_test_prefix)
        os.makedirs(indiv_configs_path, exist_ok=True)
        for vwa_task in vwa_tasks_config:
            site_mapping(vwa_task)
            with open(os.path.join(indiv_configs_path, str(vwa_task["task_id"]) + ".json"), 'w') as json_file:
                json.dump(vwa_task, json_file, indent=4)
        print("     Total", len(vwa_tasks_config), "jsons saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Separates {website}_privacy.json into single test cases")
    parser.add_argument("--combined_dir", type=str, default="wa_format/", help="path to the combined jsons folder")
    args = parser.parse_args()
    print(args)
    main(args)
