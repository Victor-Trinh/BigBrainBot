import discord
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import random
from numpy import array, log2, where

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

    realsolutions = ["real" for i in range(len(realLines))]
    fakesolutions = ["fake" for i in range(len(fakeLines))]

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
        sample = message.content.replace("$predict", "")
        v = c.transform([sample])
        result = t.predict(v)
        await message.channel.send("The headline is {}. Ready for next headline!".format(result[0]))
    if message.content.startswith('$about'):
        await message.channel.send("```I predict if US political headlines are real or fake! (with an accuracy of around 70-80%) \
            \nTry it out! \"$predict <headline>\"```")


if __name__ == "__main__":
    load_data("big_brain/clean_real.txt", "big_brain/clean_fake.txt")
    client.run('NjI5NTQ3MTcwMjczNjg5NjIw.XZbWIA.asGH2aR9rXxhBHtsfOBD_kZvglg')
    
    print("got here")
    
