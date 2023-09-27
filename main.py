from datetime import datetime

import numpy as np
import pandas as pd
import pdfkit
import PyPDF2
from jinja2 import Environment, PackageLoader, select_autoescape


def load_and_clean_users(file_name: str) -> pd.DataFrame:
    """
    Read CSV with users and check that all required fields exist.
    """
    df = pd.read_csv(file_name)
    if df.isnull().sum().sum() > 0:
        raise Exception(
            'id or ig_username or ig_num_followers is null but required'
        )
    return df


def load_and_clean_user_posts(file_name: str) -> pd.DataFrame:
    """
    Read CSV with user posts, replace null values
    and check that all required fields exist.
    """
    values = {
        'caption_text': '',
        'caption_tags': lambda: [],
        'like_count': 0,
        'comment_count': 0,
    }
    df = pd.read_csv(file_name)
    df.fillna(value=values, inplace=True)
    if df.isnull().sum().sum() > 0:
        raise Exception(
            'person_id or taken_at is null but required'
        )
    df['caption_text'] = df['caption_text'].astype('string')
    df['taken_at'] = pd.to_datetime(df['taken_at'])
    return df


def filter_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Filter df by start and end dates,
    taken_at is used as a date column.
    """
    date_format = '%Y-%m-%d'
    start_date = datetime.strptime(start, date_format)
    end_date = datetime.strptime(end, date_format)
    return df[(df['taken_at'] >= start_date) & (df['taken_at'] <= end_date)]


def filter_by_brand(df: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    """
    Filter df by mentioned tags,
    caption_text is used as a target column.
    """
    return df[df['caption_text'].apply(lambda x: any(t in x for t in tags))]


def df_to_html(df: pd.DataFrame) -> str:
    """
    Generate html from df using predefined template.
    """
    env = Environment(
        loader=PackageLoader('main'),
        autoescape=select_autoescape()
    )
    template = env.get_template('main_template.html')
    output = template.render(rows=df.to_dict(orient='records'))
    html_file = './templates/main.html'
    with open(html_file, 'w') as f:
        f.write(output)
    return html_file


def html_to_pdf(html_file: str) -> None:
    """
    Convert html to pdf and add title page.
    """
    title_page_html = 'templates/title_page.html'
    title_page_pdf = './data/title_page.pdf'
    options_title_page = {
        'quiet': '',
        'orientation': 'Landscape',
        'page-size': 'Letter',
        'margin-top': '0in',
        'margin-right': '0in',
        'margin-bottom': '0in',
        'margin-left': '0in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None,
    }
    pdfkit.from_file(
        title_page_html,
        title_page_pdf,
        options=options_title_page,
    )

    main_html = html_file
    main_pdf = 'data/main.pdf'
    options_main = {
        'quiet': '',
        'orientation': 'Landscape',
        'page-size': 'Letter',
        'margin-top': '1.1in',
        'margin-right': '0.9in',
        'margin-bottom': '1.1in',
        'margin-left': '0.9in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None,
        'header-right': 'Page [page]/[toPage]',
        'header-html': 'templates/header.html',
        'footer-html': 'templates/footer.html',
    }
    pdfkit.from_file(
        main_html,
        main_pdf,
        options=options_main
    )

    mergeFile = PyPDF2.PdfMerger()
    mergeFile.append(PyPDF2.PdfReader(title_page_pdf))
    mergeFile.append(PyPDF2.PdfReader(main_pdf))
    mergeFile.write("report.pdf")


def main() -> None:
    users_df = load_and_clean_users('./data/users.csv')
    user_posts_df = load_and_clean_user_posts('./data/user_posts.csv')

    start_date = '2021-01-01'
    end_date = '2021-02-01'
    user_posts_df = filter_by_date(user_posts_df, start_date, end_date)

    tags = ['@bubbleroom', '#bubbleroom', '#bubbleroomstyle']
    user_posts_df = filter_by_brand(user_posts_df, tags)

    user_posts_df['post_mentions'] = (
        user_posts_df['caption_text'].str.count('@bubbleroom').astype(int)
    )
    user_posts_df['post_hashtags'] = (
        user_posts_df['caption_text'].str.count('#bubbleroom').astype(int)
    )

    user_posts_df['comments_with_mention'] = np.where(
        user_posts_df['post_mentions'] != 0,
        user_posts_df['comment_count'],
        0
    )
    user_posts_df['likes_with_mention'] = np.where(
        user_posts_df['post_mentions'] != 0,
        user_posts_df['like_count'],
        0
    )

    user_posts_grouped_df = user_posts_df.groupby('person_id').agg(
        comments_all=('comment_count', 'sum'),
        comments_with_mention_all=('comments_with_mention', 'sum'),
        likes_all=('like_count', 'sum'),
        likes_with_mention_all=('likes_with_mention', 'sum'),
        posts_all=('person_id', 'size'),
        post_mentions_all=('post_mentions', 'sum'),
        post_hashtags_all=('post_hashtags', 'sum'),
    )
    result_df = users_df.merge(
        user_posts_grouped_df,
        left_on='id',
        right_on='person_id'
    )

    result_df['engagement_general'] = (
        ((result_df['likes_all']
          + result_df['comments_all'])
         / result_df['ig_num_followers'] * 100).round(2)
    )
    result_df['engagement_specific'] = (
        ((result_df['likes_with_mention_all']
          + result_df['comments_with_mention_all'])
         / result_df['ig_num_followers'] * 100).round(2))

    result_df['score'] = (
            result_df['posts_all']
            + result_df['post_mentions_all']
            + result_df['post_hashtags_all']
    )
    result_df.sort_values(by=['score'], ascending=False, inplace=True)

    html_to_pdf(df_to_html(result_df))


if __name__ == '__main__':
    main()
