import pandas as pd
lines_filepath = "movie_lines.txt"
conv_filepath = "movie_conversations.txt"
line_fields = ["lineID","characterID","movieID","character","text"]
lines = {}

with open(lines_filepath, 'r', encoding='iso-8859-1') as f:
    for line in f:
        values = line.split(" +++$+++ ")
        # Extract fields
        lineObj = {}
        for i, field in enumerate(line_fields):
            lineObj[field] = values[i]
        lines[lineObj['lineID']] = lineObj

conv_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
conversations = []

with open(conv_filepath, 'r', encoding='iso-8859-1') as f:
    for line in f:
        values = line.split(" +++$+++ ")
        convObj = {}
        for i, field in enumerate(conv_fields):
            convObj[field] = values[i]
        lineIds = eval(convObj["utteranceIDs"])
        convObj["lines"] = []
        for lineId in lineIds:
            convObj["lines"].append(lines[lineId])
        conversations.append(convObj)

qa_pairs = []
x = []
y = []
for conversation in conversations:
    for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
        inputLine = conversation["lines"][i]["text"].strip()
        targetLine = conversation["lines"][i+1]["text"].strip()
        if inputLine and targetLine:
            qa_pairs.append([inputLine, targetLine])
            x.append(inputLine)
            y.append(targetLine)



dic = {'x':x,
       'y':y}
df = pd.DataFrame(dic)
df.to_csv("train.csv")
#print(len(qa_pairs))