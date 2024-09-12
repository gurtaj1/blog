
# POC: Moderation of User Generated Content using Microsoft Azure Content Safety

## Why Microsoft Azure

This project is company which already work closely with Microsoft and Azure. It makes sense to see how their product works before we begin to compare to others.

## Getting familiar with Microsoft Azure Content Safety

Azure have Content Safety Studio which is essentially a little playground to be able to test and playwith their Content Safety Features. It also provides you with an endpoint and an API key to authenticate our requests.

The endpoint is essentially the way in which we send our content and request that it be assessed for safety (content moderation).

![https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/content_safety_studio.png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/content_safety_studio.png)

![https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/content_safety_text_recognition_try_out.png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/content_safety_text_recognition_try_out.png)

## Setting up the backend

Since we already have access to AWS I decided to use their services. I created a lambda to execute all the logic which can be seen below (this was writen in Python):

```python
import enum
import json
import requests
import os
from typing import Union

class MediaType(enum.Enum):
    Text = 1
    Image = 2

class Category(enum.Enum):
    Hate = 1
    SelfHarm = 2
    Sexual = 3
    Violence = 4

class Action(enum.Enum):
    Accept = 1
    Reject = 2

class DetectionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return f"DetectionError(code={self.code}, message={self.message})"

class Decision(object):
    def __init__(
        self, suggested_action: Action, action_by_category: dict[Category, Action]
    ) -> None:
        self.suggested_action = suggested_action
        self.action_by_category = action_by_category

class ContentSafety(object):
    def __init__(self, endpoint: str, subscription_key: str, api_version: str) -> None:
        self.endpoint = endpoint
        self.subscription_key = subscription_key
        self.api_version = api_version

    def build_url(self, media_type: MediaType) -> str:
        if media_type == MediaType.Text:
            return f"{self.endpoint}/contentsafety/text:analyze?api-version={self.api_version}"
        elif media_type == MediaType.Image:
            return f"{self.endpoint}/contentsafety/image:analyze?api-version={self.api_version}"
        else:
            raise ValueError(f"Invalid Media Type {media_type}")

    def build_headers(self) -> dict[str, str]:
        return {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/json",
        }

    def build_request_body(
        self,
        media_type: MediaType,
        content: str,
        blocklists: list[str],
    ) -> dict:
        if media_type == MediaType.Text:
            return {
                "text": content,
                "blocklistNames": blocklists,
            }
        elif media_type == MediaType.Image:
            return {"image": {"content": content}}
        else:
            raise ValueError(f"Invalid Media Type {media_type}")

    def detect(
        self,
        media_type: MediaType,
        content: str,
        blocklists: list[str] = [],
    ) -> dict:
        url = self.build_url(media_type)
        headers = self.build_headers()
        request_body = self.build_request_body(media_type, content, blocklists)
        payload = json.dumps(request_body)

        response = requests.post(url, headers=headers, data=payload)

        res_content = response.json()

        if response.status_code != 200:
            raise DetectionError(
                res_content["error"]["code"], res_content["error"]["message"]
            )

        return res_content

    def get_detect_result_by_category(
        self, category: Category, detect_result: dict
    ) -> Union[int, None]:
        category_res = detect_result.get("categoriesAnalysis", None)
        for res in category_res:
            if category.name == res.get("category", None):
                return res
        raise ValueError(f"Invalid Category {category}")

    def make_decision(
        self,
        detection_result: dict,
        reject_thresholds: dict[Category, int],
    ) -> Decision:
        action_result = {}
        final_action = Action.Accept
        for category, threshold in reject_thresholds.items():
            if threshold not in (-1, 0, 2, 4, 6):
                raise ValueError("RejectThreshold can only be in (-1, 0, 2, 4, 6)")

            cate_detect_res = self.get_detect_result_by_category(
                category, detection_result
            )
            if cate_detect_res is None or "severity" not in cate_detect_res:
                raise ValueError(f"Can not find detection result for {category}")

            severity = cate_detect_res["severity"]
            action = (
                Action.Reject
                if threshold != -1 and severity >= threshold
                else Action.Accept
            )
            action_result[category] = action
            if action.value > final_action.value:
                final_action = action

        if (
            "blocklistsMatch" in detection_result
            and detection_result["blocklistsMatch"]
            and len(detection_result["blocklistsMatch"]) > 0
        ):
            final_action = Action.Reject

        return Decision(final_action, action_result)

def lambda_handler(event, context):
    options = event.get('options', {})
    reject_threshold = int(options.get('reject_threshold', 4))
    media_type = MediaType.Image if options.get('media_type') == 'Image' else MediaType.Text

    endpoint = "https://guide-contentsafety-poc.cognitiveservices.azure.com/"
    subscription_key = os.environ['API_KEY']
    api_version = "2023-10-01"

    content_safety = ContentSafety(endpoint, subscription_key, api_version)

    blocklists = []

    content = event.get('content', '')

    detection_result = content_safety.detect(media_type, content, blocklists)

    reject_thresholds = {
        Category.Hate: reject_threshold,
        Category.SelfHarm: reject_threshold,
        Category.Sexual: reject_threshold,
        Category.Violence: reject_threshold,
    }

    decision_result = content_safety.make_decision(detection_result, reject_thresholds)
    
    # Mapping from original category keys to desired output strings
    category_map = {
        Category.Hate: "Hate",
        Category.SelfHarm: "Self Harm",
        Category.Sexual: "Sexual",
        Category.Violence: "Violence"
    }

    # Iterate over the action_by_category dictionary
    rejected_categories = []
    for category, action in decision_result.action_by_category.items():
        if action != Action.Accept:
            rejected_categories.append(category_map[category])

    return {
        'statusCode': 200,
        'body': json.dumps({
            'suggested_action': decision_result.suggested_action.name,
            "rejected_categories": rejected_categories,
            'action_by_category': {str(k): v.name for k, v in decision_result.action_by_category.items()},
            "reject_threshold": reject_threshold,
            "media_type": str(media_type),
        })
    }

```

To trigger this logic I created an endpoint that would pass the relevant information (content, content type, and reject thresholds) to the lambda.
As can be seen, I used the lambda apply logic dinamically, dependent on content type. I then made sure content was interpretable by Azure Content Safety before sending it off to their endpoint, along with the relevant variables. The response was then manipulated in a way that was desirable for frontend consumption.

## Applying the Frontend Logic

This was all done in javascript, within the React framework, as this is what the platform is built with. It was simple enough to add a step between our current logic (user hitting post and the content actually being posted, or user hitting save on their profile and the profile actually being saved). The content would simply be sent to the endpoint that would trigger our lambda. Then, once a response is received, we would either submit the content right away, or display a modal that would warn the user that their content has not passed content safety checks.

The execution of this logic can be seen in the two videos below:

![https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/content_safety-new_post.gif](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/content_safety-new_post.gif)

![https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/content_safety-user_bio.gif](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/content_safety-user_bio.gif)

## And that was it

I had officially made the POC for using Microsoft Azure Content Safety on our platform to moderate user generated content. The Next step will now be to create another POC for the next biggest competitor in terms of AI solutions and see how their service compares.
