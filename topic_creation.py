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
selected_topics = []


def topic_list():
    return topic

def subtopic_select(topic):
    from quiz_creation_page import quiz_creation_page
    sub_topics = []
    for topic in selected_topics:
        if topic == language:
            sub_topics.extend(language_topic)
        elif topic == mathematic:
            sub_topics.extend(mathematic_topic)
        elif topic == social_science:
            sub_topics.extend(social_science_topic)
        elif topic == natural_science:
            sub_topics.extend(natural_science_topic)
        elif topic == humanity:
            sub_topics.extend(humanity_topic)
        elif topic == engineering:
            sub_topics.extend(engineering_topic)
        elif topic == art:
            sub_topics.extend(art_topic)
        return sub_topics




