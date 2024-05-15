from typing import List
import streamlit as st
from quiz_creation_page import quiz_creation_page


    # 퀴즈 주제 선택
language = "언어"
english = "영어"
korean = "한국어"
language_topic = [english, korean]

mathematic = "수리"
algebra = "대수학"
geometry = "기하학"
calculus = "미적분학"
statistics = "통계학"
mathematic_topic = [algebra, geometry, calculus, statistics]

social_science = "사회과학"
psychology = "심리학"
sociology = "사회학"
economics = "경제학"
political_science = "정치학"
social_science_topic = [psychology, sociology, economics, political_science]

natural_science = "자연과학"
physics = "물리학"
chemistry = "화학"
biology = "생물학"
astronomy = "천문학"
natural_science_topic = [physics, chemistry, biology, astronomy]

humanity = "인문학"
philosophy = "철학"
history = "역사학"
literature = "문학"
art_history = "미술사"
humanity_topic = [philosophy, history, literature, art_history]

engineering = "공학"
computer_engineering = "컴퓨터 공학"
architectural_engineering = "건축공학"
engineering_topic = [computer_engineering, architectural_engineering]

art = "예술"
# western_art = "서양화"
# oriental_art = "동양화"
film = "영화"
novel = "소설"
art_topic = [film,
             # western_art,
             # oriental_art,
             novel]

topic = [language, mathematic, social_science, natural_science,
         humanity, engineering, art]
now_topic = [language + "(미지원)", mathematic + "(미지원)",
             social_science + "(미지원)", natural_science + "(미지원)",
             humanity + "(미지원)", engineering + "(미지원)",
             art + "(영화 분야 지원)"]
selected_topics = []


def topic_select():
    return now_topic

def topic_list():
    return topic

def subtopic_select(a, b):
    for i in range(len(selected_topics)):
        if topic == language:
            subtopics = language_topic
        elif topic == mathematic:
            subtopics = mathematic_topic
        elif topic == social_science:
            subtopics = social_science_topic
        elif topic == natural_science:
            subtopics = natural_science_topic
        elif topic == humanity:
            subtopics = humanity_topic
        elif topic == engineering:
            subtopics = engineering_topic
        elif topic == art:
            subtopics = art_topic
        return sub_topic




