# importing modules
import streamlit as st
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup


def generate_amazon_link(search_query):
    """
    Generates and returns an amazon search link based on search query for data scraping

    :param search_query: string of specifications of a laptop
    :return: an amazon search link
    """
    # adding search query to base URL and returning
    return "https://www.amazon.in/s?k=laptop" + search_query


def get_item_links(search_query):
    """
    Scrapes data from the search page and prints image and amazon links of top three laptops on the page

    :param search_query: string of specifications of a laptop
    """

    # header contains User Agent which is to make sure that the website responds to the request thinking
    # that the request is from a real browser(which it is )
    headers = {
        'Referer': 'https://www.google.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    # gets content of the webpage (get request)
    response = requests.get(generate_amazon_link(search_query), headers=headers)
    r= requests.get('https://httpbin.org/headers')
    st.write(r.text)
    print(response)
    # creating BeautifulSoup object to read the data of the webpage
    soup = BeautifulSoup(response.content, 'html.parser')

    # getting all the links present on the webpage (first links are the links of the products )
    links = soup.find_all('a', attrs={
        'class': 'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal'})

    # getting titles of all products (inspect the webpage you are scraping)
    titles = soup.find_all('span', attrs={'class': 'a-size-medium a-color-base a-text-normal'})

    # getting images of all products
    images = soup.find_all('img', attrs={'class': 's-image'})

    # printing the top three products with images and links to amazon
    for i in range(2,5):
        # extracting text from each title html
        title = titles[i].text

        # extracting image url from each url
        url = 'https://www.amazon.in' + links[i].get('href')

        # columns used to print text next to image and not below the image
        column = st.columns([1, 2])  # Adjust the width ratios as needed

        # printing image
        column[0].image(images[i].get('src'), width=200)

        # printing title with hyperlink
        column[1].write(f"[{title}]({url})")
    st.write(r.text)

# importing the ML model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# importing the dataframe
df = pickle.load(open('df.pkl', 'rb'))

# Heading of the webapp
st.markdown("## Laptop Predictor and Recommender")

# making 3 columns for features
columns = st.columns([1, 1, 1])

# getting brand from user through a select box
company = columns[0].selectbox('Brand', df['Company'].unique())

# getting type of laptop from user through a select box
type = columns[1].selectbox('Type', df['Type'].unique())

# getting RAM from user through a select box
ram = columns[2].selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# getting weight from user through an input box
weight = columns[0].number_input('Weight', value=2.0)

# Touchscreen or not
touchscreen = columns[1].selectbox('Touchscreen', ['No', 'Yes'])

# IPS or not
ips = columns[2].selectbox('IPS', ['No', 'Yes'])

# getting screen size from user through an input box
screen_size = columns[0].number_input('Screen Size', value=15.6, )

# getting resolution from user through a select box
resolution = columns[1].selectbox('Screen Resolution',
                                  ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800',
                                   '2560x1600',
                                   '2560x1440', '2304x1440'])

# getting cpu input from user through a select box
cpu = columns[2].selectbox('CPU', df['CPU'].unique())

# getting gpu input from user through a select box
gpu = columns[0].selectbox('GPU', df['GPU'].unique())

# getting hdd input from user through a select box
hdd = columns[1].selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# getting ssd input from user through a select box
ssd = columns[2].selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# getting Operating System input from user through a select box
os = columns[0].selectbox('Operating System', df['OS'].unique())

# creating a button and what to do if its pressed
if st.button('Predict Price'):
    # ppi(pixels per inch) will be calculated based on user inputs
    ppi = None

    # changing user inputs YES and NO to 1 and 0
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # calculating ppi
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # creating data for feeding it to the model
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, gpu, hdd, ssd, os])
    query = query.reshape(1, 12)

    # feeding data and predicting price
    st.markdown("### The predicted price of this configuration is Rs." + str(int(np.exp(pipe.predict(query)[0]))))

    # printing empty space
    st.text("")

    # printing a faded line
    st.markdown("---")

    # printing similar laptops
    st.markdown("### Similar laptops on Amazon")

    # If company is apple change search query to Apple Macbook and call the get item links function
    if company == 'Apple':
        get_item_links(f'Apple+Macbook+{ram}+GB+RAM')
    else:
        get_item_links(f'{company}+{type}+{ssd}+SSD+{ram}+GB+RAM+{cpu}+{gpu}')
