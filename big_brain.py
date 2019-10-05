import discord
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import random


c = CountVectorizer()
client = discord.Client()
t = DecisionTreeClassifier(max_depth=90)

def load_data(fileName1, fileName2):

    f1 = open(fileName1)
    f2 = open(fileName2)
    
    realLines = []
    line = f1.readline()
    while line:
        realLines.append(line.strip("\n"))
        line = f1.readline()
    
    fakeLines = []
    line = f2.readline()
    while line:
        fakeLines.append(line.strip("\n"))
        line=f2.readline()

    realsolutions = ["REAL" for i in range(len(realLines))]
    fakesolutions = ["FAKE" for i in range(len(fakeLines))]

    train_vecs = c.fit_transform(realLines + fakeLines)
    t.fit(train_vecs, realsolutions + fakesolutions)

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    if message.content.startswith('$predict'):
        if message.content == 'predict':
            await message.channel.send("Send a headline! \"$predict <headline>\"")

        sample = message.content.replace("$predict", "")
        v = c.transform([sample])
        result = t.predict(v)
        emote = ":negative_squared_cross_mark:" if result == "FAKE" else ":white_check_mark:"
        m = "This headline is {} {}. \n Ready for next headline!".format(result[0], emote)
        await message.channel.send(m)

    if message.content.startswith('$about'):
        await message.channel.send("```I predict if US political headlines are real or fake! (with an accuracy of around 70-80%) \
            \nTry it out! \"$predict <headline>\"```")


if __name__ == "__main__":
    load_data("clean_real.txt", "clean_fake.txt")
    # https://www.kaggle.com/mrisdal/fake-news/data
    # https://www.kaggle.com/therohk/million-headlines 
    client.run('PLACEHOLDER')
    
    print("Exiting.")
    
