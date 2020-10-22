import json


# Find the dataset and initialize it for the dataset and dataloader
def initializeNQDataset(dataset_location, tokenizer):
    filedataset = open(dataset_location, "r")
    contexts = []
    questions = []
    answers = []
    counter = 0
    for line in filedataset:
        # Set the number of datapoints used or skip if entire dataset should be used
        if counter == 100:
            break
        data = json.loads(line)
        sq = {}
        for annotation in data["annotations"]:
            sq = annotation["short_answers"]
        if sq:
            answers.append(sq)
            contexts.append(data["document_text"])
            questions.append(data["question_text"])
            counter = counter + 1

    # Tokenize the contexts and questions
    # Change max_length accordingly to the model used
    encodings = tokenizer(
        contexts, questions, truncation=True, padding=True, max_length=512
    )
    contexts = []
    questions = []

    # Find positions and append them to encodings
    start_positions = []
    end_positions = []
    for answer in answers:
        start_positions.append(answer[0]["start_token"])
        end_positions.append(answer[0]["end_token"])
    encodings.update(
        {"start_positions": start_positions, "end_positions": end_positions}
    )
    answers = []
    start_positions = []
    end_positions = []

    return encodings
