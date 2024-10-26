import tweepy
import openai
import schedule
import time
import os
import logging
from dotenv import load_dotenv
import random
import json
from difflib import SequenceMatcher
import ntplib
from textblob import TextBlob
from functools import wraps
import queue
import threading
import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from collections import deque

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the config directory
CONFIG_DIR = PROJECT_ROOT / 'config'

# Search for any .env file in the config directory
env_file = next(CONFIG_DIR.glob('*.env'), None)

if env_file:
    print(f"Found environment file: {env_file}")
    # Load the environment file
    load_dotenv(env_file)
else:
    print("No .env file found in the config directory.")
    # You might want to raise an exception here or set default values

# Now you can access environment variables using os.getenv
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CUSTOM_TWEETS_FILE = os.getenv('CUSTOM_TWEETS_FILE')
POSTED_TWEETS_FILE = os.getenv('POSTED_TWEETS_FILE')
ENGAGEMENT_FILE = os.getenv('ENGAGEMENT_FILE')
SENTIMENT_FILE = os.getenv('SENTIMENT_FILE')

# Now you can set up logging
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "twitter_bot.log")
    my_handler = RotatingFileHandler(log_file, mode='a', maxBytes=5*1024*1024, 
                                     backupCount=2, encoding=None, delay=0)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO)
    logger.addHandler(my_handler)
    
    return logger

# Set up logger
logger = setup_logging()

# Print current working directory
current_dir = os.getcwd()
logger.info(f"Current working directory: {current_dir}")

# Possible .env file locations
env_locations = [
    PROJECT_ROOT / '.env',
    PROJECT_ROOT / 'twitter_automation_bot' / '.env',
    PROJECT_ROOT / 'twitter_automation_bot' / 'config' / '.env',
    PROJECT_ROOT / 'eaccmaxi.env',
    Path(os.getenv('TWITTER_BOT_ENV_FILE', ''))
]

# Try to load the .env file from different locations
env_loaded = False
for env_path in env_locations:
    if env_path.exists():
        logger.info(f".env file found at {env_path}")
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Dotenv loaded from {env_path}")
        env_loaded = True
        break
    else:
        logger.debug(f".env file not found at {env_path}")

if not env_loaded:
    logger.error("No .env file found in any of the expected locations.")
    logger.info("Please ensure your .env file is in one of the following locations:")
    for location in env_locations:
        logger.info(f"- {location}")
    logger.info("Or set the TWITTER_BOT_ENV_FILE environment variable to specify the location.")

# Global variables for file paths
CUSTOM_TWEETS_FILE = None
POSTED_TWEETS_FILE = None
ENGAGEMENT_FILE = None
SENTIMENT_FILE = None
LOG_DIR = None
LOG_FILE = None
DATA_DIR = None

# NTP time synchronization
def sync_time():
    ntp_client = ntplib.NTPClient()
    try:
        response = ntp_client.request('pool.ntp.org')
        time_diff = response.tx_time - time.time()
        logger.info(f"Time difference with NTP server: {time_diff:.2f} seconds")
    except:
        logger.warning("Unable to sync with NTP server. Using system time.")

sync_time()

# Twitter API v2 setup
def initialize_twitter_client(api_key, api_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    client = tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
        wait_on_rate_limit=True
    )
    return client, api

def initialize_openai(api_key):
    openai.api_key = api_key

def load_config():
    global CUSTOM_TWEETS_FILE, POSTED_TWEETS_FILE, ENGAGEMENT_FILE, SENTIMENT_FILE, LOG_DIR, LOG_FILE, DATA_DIR

    config = {
        # API keys
        'TWITTER_API_KEY': os.getenv('TWITTER_API_KEY'),
        'TWITTER_API_SECRET': os.getenv('TWITTER_API_SECRET'),
        'TWITTER_ACCESS_TOKEN': os.getenv('TWITTER_ACCESS_TOKEN'),
        'TWITTER_ACCESS_TOKEN_SECRET': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        
        # Other configuration variables
        'CUSTOM_TWEETS_FILE': os.getenv('CUSTOM_TWEETS_FILE'),
        'POSTED_TWEETS_FILE': os.getenv('POSTED_TWEETS_FILE'),
        'ENGAGEMENT_FILE': os.getenv('ENGAGEMENT_FILE'),
        'SENTIMENT_FILE': os.getenv('SENTIMENT_FILE'),
    }
    
    # Now set the other file paths
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    LOG_FILE = os.path.join(LOG_DIR, 'twitter_bot.log')
    CUSTOM_TWEETS_FILE = os.path.join(DATA_DIR, config['CUSTOM_TWEETS_FILE'] or 'custom_tweets.txt')
    POSTED_TWEETS_FILE = os.path.join(DATA_DIR, config['POSTED_TWEETS_FILE'] or 'posted_tweets.json')
    ENGAGEMENT_FILE = os.path.join(DATA_DIR, config['ENGAGEMENT_FILE'] or 'tweet_engagement.json')
    SENTIMENT_FILE = os.path.join(DATA_DIR, config['SENTIMENT_FILE'] or 'tweet_sentiment.json')
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    logger.info("Loaded configuration:")
    for key, value in config.items():
        if value:
            logger.info(f"{key}: Set (length: {len(value)})")
        else:
            logger.info(f"{key}: Not found or empty")
    
    # Check for missing API keys
    api_keys = ['TWITTER_API_KEY', 'TWITTER_API_SECRET', 'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_TOKEN_SECRET', 'OPENAI_API_KEY']
    missing_keys = [key for key in api_keys if not config[key]]
    if missing_keys:
        raise ValueError(f"The following API keys are missing or empty: {', '.join(missing_keys)}. Please check your .env file.")
    
    return config

# Global variables that will be initialized in main()
client = None
post_cap_tracker = None

# List of prompts
prompts = [
    "Generate a tweet that highlights how fast technology is changing the world.",
    "Compose a tweet about the speed of innovation and how accelerationism is shaping the future.",
    "Write a tweet about AI transforming our daily lives in a fast and impactful way.",
    "Create a tweet that captures the essence of rapid technological advancement.",
    "Generate a tweet that challenges people to think about the pace of progress in society.",
    "Generate a fun and humorous tweet about robots becoming part of everyday life.",
    "Compose a witty tweet about how AI will do mundane tasks better than humans.",
    "Write a funny tweet about the future of automation and how it affects us.",
    "Create a humorous tweet comparing today's tech to a futuristic, sci-fi reality.",
    "Generate a tweet that humorously imagines a future run by AI and robots.",
    "Draft an informative tweet explaining how accelerationism could transform the future of work.",
    "Generate an educational tweet about how AI and automation are reshaping industries.",
    "Write a tweet that explains accelerationism in simple terms and its impact on society.",
    "Create a tweet that breaks down how accelerationism can drive technological progress in healthcare.",
    "Generate a tweet discussing the role of accelerationism in advancing space exploration.",
    "Write a thought-provoking tweet asking how people feel about the impact of AI on their lives.",
    "Generate a tweet asking followers what they think about the speed of innovation and where it's taking us.",
    "Compose a tweet that asks followers if they believe accelerationism will lead to a utopian or dystopian future.",
    "Draft a tweet asking if AI will benefit everyone equally or only the privileged few.",
    "Create a tweet asking how people think automation will change the job market in the next decade.",
    "Generate a tweet that makes people reflect on the ethical implications of accelerationism and AI.",
    "Write a tweet that invites followers to think about the risks and rewards of rapid technological progress.",
    "Compose a reflective tweet about how human values are affected by accelerationist ideologies.",
    "Draft a tweet that challenges followers to consider how technology might evolve faster than we can control.",
    "Generate a tweet reflecting on how societies must adapt to the speed of innovation.",
    "Compose a pop culture-inspired tweet about how futuristic tech ideas are becoming reality.",
    "Generate a tweet referencing a sci-fi movie or series to illustrate how fast tech is evolving.",
    "Write a tweet comparing the current state of AI to a funny sci-fi scenario.",
    "Create a playful tweet about living in a future filled with drones, robots, and AI.",
    "Generate a tweet that mixes pop culture and accelerationism to engage a tech-savvy audience.",
    "Write a motivational tweet that inspires people to embrace rapid tech progress.",
    "Generate an inspiring tweet about how accelerationism can create a better future for everyone.",
    "Compose a tweet that encourages people to think big and act fast in the face of technological change.",
    "Draft a tweet that motivates followers to see the potential of accelerationism to solve global problems.",
    "Create a tweet that inspires innovation and highlights the importance of staying ahead in the tech race.",
    "Generate a tweet asking how tech will redefine human relationships in the future.",
    "Compose a tweet asking how automation and AI will change our sense of identity.",
    "Write a tweet asking whether technology brings us closer together or drives us apart.",
    "Create a tweet that asks followers how they think the rapid pace of technology affects global economies.",
    "Generate a tweet asking how societies can balance tech innovation with preserving human values.",
    "Write a tweet challenging the notion that all technological progress is positive.",
    "Generate a tweet questioning whether accelerationism is moving too fast for humanity to keep up.",
    "Compose a tweet challenging people to consider the downsides of rapid tech adoption.",
    "Draft a tweet that critiques the unequal distribution of benefits from technological acceleration.",
    "Create a tweet that questions whether our ethical frameworks can keep up with accelerating technologies."
]

# Rate limiters
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            self.calls = [call for call in self.calls if call > now - self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] - (now - self.period)
                time.sleep(sleep_time)
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper

app_rate_limiter = RateLimiter(max_calls=450, period=900)  # 450 requests per 15 minutes for app-level
user_rate_limiter = RateLimiter(max_calls=900, period=900)  # 900 requests per 15 minutes for user-level

# Post cap tracker
class PostCapTracker:
    def __init__(self, monthly_cap):
        self.monthly_cap = monthly_cap
        self.current_month = datetime.datetime.now().month
        self.post_count = 0

    def can_make_request(self):
        current_month = datetime.datetime.now().month
        if current_month != self.current_month:
            self.current_month = current_month
            self.post_count = 0
        return self.post_count < self.monthly_cap

    def increment_count(self, count=1):
        self.post_count += count

# Caching decorator
def cache_with_timeout(timeout=300):  # 5 minutes default
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < timeout:
                    return result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator

# Exponential backoff
def exponential_backoff(attempt, max_attempts=5, base_delay=1, max_delay=60):
    if attempt < max_attempts:
        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.1 * base_delay), max_delay)
        time.sleep(delay)
        return True
    return False

# Queue system for handling requests
request_queue = queue.Queue()

def worker():
    while True:
        func, args, kwargs = request_queue.get()
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error processing request: {e}")
        finally:
            request_queue.task_done()

# Start worker threads
for _ in range(5):  # Adjust number of threads as needed
    threading.Thread(target=worker, daemon=True).start()

def enqueue_request(func, *args, **kwargs):
    request_queue.put((func, args, kwargs))

@app_rate_limiter
@cache_with_timeout(timeout=300)
def generate_content(prompt, temperature=0.7):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates tweet content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=280,
        n=1,
        stop=None,
        temperature=temperature,
    )
    # Handle potential changes in API response structure
    if isinstance(response.choices[0], dict):
        return response.choices[0].get('message', {}).get('content', '').strip()
    else:
        return response.choices[0].message.content.strip()

def is_similar(new_tweet, posted_tweets, threshold=0.7):
    for tweet in posted_tweets:
        similarity = SequenceMatcher(None, new_tweet.lower(), tweet.lower()).ratio()
        if similarity > threshold:
            return True
    return False

def get_unique_content(prompt):
    posted_tweets = load_posted_tweets()
    max_attempts = 5
    for _ in range(max_attempts):
        content = generate_content(prompt)
        if content and not is_similar(content, posted_tweets):
            return content
    logger.warning("Failed to generate unique content after multiple attempts.")
    return None

@user_rate_limiter
def post_tweet(content):
    if not post_cap_tracker.can_make_request():
        logger.warning("Monthly post cap reached. Skipping tweet.")
        return None

    try:
        response = client.create_tweet(text=content)
        tweet_id = response.data['id']
        logger.info(f"Tweet posted successfully: {content}")
        save_posted_tweet(content)
        post_cap_tracker.increment_count()
        return tweet_id
    except Exception as e:
        logger.error(f"Error posting tweet: {str(e)}")
        return None

def get_custom_tweet():
    logger.info("Attempting to get custom tweet")
    try:
        with open(CUSTOM_TWEETS_FILE, 'r') as f:
            tweets = f.readlines()
        if tweets:
            tweet = random.choice(tweets).strip()
            logger.info(f"Custom tweet selected: {tweet}")
            
            # Remove the used tweet from the file
            tweets.remove(tweet + '\n')
            with open(CUSTOM_TWEETS_FILE, 'w') as f:
                f.writelines(tweets)
            logger.info("Used tweet removed from custom tweets file")
            
            return tweet
        else:
            logger.info("No custom tweets available")
            return None
    except Exception as e:
        logger.exception(f"Error reading custom tweets: {str(e)}")
        return None

@cache_with_timeout(timeout=3600)  # Cache for 1 hour
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

@app_rate_limiter
def track_engagement(tweet_id):
    try:
        tweet = client.get_tweet(tweet_id, tweet_fields=['public_metrics'])
        public_metrics = tweet.data.public_metrics
        engagement = {
            "retweets": public_metrics['retweet_count'],
            "likes": public_metrics['like_count'],
            "replies": public_metrics['reply_count']
        }
        
        engagement_data = load_engagement_data()
        if str(tweet_id) not in engagement_data:
            engagement_data[str(tweet_id)] = []
        engagement_data[str(tweet_id)].append({
            "timestamp": time.time(),
            "engagement": engagement
        })
        
        save_engagement_data(tweet_id, engagement)
        
        logger.info(f"Engagement for tweet {tweet_id}: {engagement}")
        return engagement
    except Exception as e:
        logger.error(f"Error tracking engagement: {str(e)}")
        return None

def load_posted_tweets():
    if os.path.exists(POSTED_TWEETS_FILE):
        with open(POSTED_TWEETS_FILE, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                logger.warning(f"Error decoding {POSTED_TWEETS_FILE}. Returning empty list.")
                return []
    return []

def save_posted_tweet(tweet):
    posted_tweets = load_posted_tweets()
    posted_tweets.append(tweet)
    with open(POSTED_TWEETS_FILE, 'w') as file:
        json.dump(posted_tweets, file)

def load_engagement_data():
    if os.path.exists(ENGAGEMENT_FILE):
        with open(ENGAGEMENT_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_engagement_data(tweet_id, data):
    engagement_data = load_engagement_data()
    if str(tweet_id) not in engagement_data:
        engagement_data[str(tweet_id)] = []
    engagement_data[str(tweet_id)].append({
        "timestamp": time.time(),
        "engagement": data
    })
    with open(ENGAGEMENT_FILE, 'w') as file:
        json.dump(engagement_data, file)

def load_sentiment_data():
    if os.path.exists(SENTIMENT_FILE):
        with open(SENTIMENT_FILE, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                logger.warning(f"Error decoding {SENTIMENT_FILE}. Returning empty dict.")
                return {}
    return {}

def save_sentiment_data(tweet_id, sentiment):
    try:
        sentiment_data = load_sentiment_data()
        sentiment_data[tweet_id] = {
            'timestamp': time.time(),
            'sentiment': sentiment
        }
        with open(SENTIMENT_FILE, 'w') as f:
            json.dump(sentiment_data, f)
        logger.info(f"Sentiment data saved for tweet {tweet_id}")
    except Exception as e:
        logger.error(f"Error saving sentiment data: {str(e)}")

class PromptRotator:
    def __init__(self, prompts):
        self.all_prompts = prompts
        self.unused_prompts = deque(random.sample(prompts, len(prompts)))
        self.used_prompts = deque(maxlen=len(prompts) // 2)  # Store half of the total prompts as used

    def get_prompt(self):
        if not self.unused_prompts:
            # If all prompts have been used, refresh the unused_prompts
            self.refresh_prompts()
        
        prompt = self.unused_prompts.popleft()
        self.used_prompts.append(prompt)
        return prompt

    def refresh_prompts(self):
        # Move some used prompts back to unused, and add new ones if available
        refresh_count = min(len(self.used_prompts), len(self.all_prompts) // 2)
        self.unused_prompts.extend(random.sample(list(self.used_prompts), refresh_count))
        self.used_prompts = deque(set(self.used_prompts) - set(self.unused_prompts), maxlen=len(self.all_prompts) // 2)
        
        # If we still need more prompts, add randomly from all_prompts
        while len(self.unused_prompts) < len(self.all_prompts) // 2:
            new_prompt = random.choice(self.all_prompts)
            if new_prompt not in self.unused_prompts and new_prompt not in self.used_prompts:
                self.unused_prompts.append(new_prompt)

    def add_prompt(self, new_prompt):
        if new_prompt not in self.all_prompts:
            self.all_prompts.append(new_prompt)
            if len(self.unused_prompts) < len(self.all_prompts) // 2:
                self.unused_prompts.append(new_prompt)

# Initialize the PromptRotator with your prompts
prompt_rotator = PromptRotator(prompts)

# Modify the job function to use PromptRotator
def job():
    logger.info("Starting job execution")
    tweet_posted = False
    try:
        content = get_custom_tweet()
        if content:
            logger.info(f"Using custom tweet: {content}")
        else:
            logger.info("No custom tweet available, generating new content")
            selected_prompt = prompt_rotator.get_prompt()
            content = get_unique_content(selected_prompt)
            logger.info(f"Generated tweet using prompt: {selected_prompt}")
            logger.info(f"Generated tweet: {content}")
        
        if content:
            image_folder = os.path.join(PROJECT_ROOT, 'data', 'images')
            image_path = get_random_image(image_folder)
            if image_path:
                logger.info(f"Selected image: {image_path}")
            else:
                logger.info("No image selected")
            
            tweet_id = post_tweet(content, image_path)
            if tweet_id:
                logger.info(f"Tweet posted successfully with ID: {tweet_id}")
                tweet_posted = True
                
                # Attempt to track engagement and save sentiment data
                try:
                    track_engagement(tweet_id)
                except Exception as e:
                    logger.error(f"Error tracking engagement: {str(e)}")
                
                try:
                    sentiment = analyze_sentiment(content)
                    save_sentiment_data(tweet_id, sentiment)
                except Exception as e:
                    logger.error(f"Error saving sentiment data: {str(e)}")
            else:
                logger.warning("Failed to post tweet")
        else:
            logger.warning("Failed to get content for tweet")
    except Exception as e:
        logger.exception(f"Error in job execution: {str(e)}")
    finally:
        logger.info("Job execution completed")
        return tweet_posted

def schedule_random_job():
    random_minute = random.randint(0, 59)
    schedule.every().hour.at(f":{random_minute:02d}").do(job).tag('random_job')

def implement_future_improvements():
    # Enhanced post_tweet function with exponential backoff
    def enhanced_post_tweet(content, image_path=None):
        attempt = 0
        while True:
            try:
                if image_path:
                    media = upload_media(image_path)
                    if media:
                        response = client.create_tweet(text=content, media_ids=[media.media_id])
                    else:
                        logger.error("Failed to upload media. Posting tweet without image.")
                        response = client.create_tweet(text=content)
                else:
                    response = client.create_tweet(text=content)
                tweet_id = response.data['id']
                logger.info(f"Tweet posted successfully: {content}")
                save_posted_tweet(content)
                return tweet_id
            except tweepy.errors.TooManyRequests:
                if not exponential_backoff(attempt):
                    logger.error("Max retries reached. Failed to post tweet.")
                    return None
                attempt += 1
            except tweepy.errors.Forbidden as e:
                logger.error(f"Forbidden error: {str(e)}")
                if "duplicate content" in str(e).lower():
                    new_prompt = prompt_rotator.get_prompt()
                    content = get_unique_content(new_prompt)
                    logger.info(f"Generated new content using prompt: {new_prompt}")
                    continue
                return None
            except Exception as e:
                logger.error(f"Unexpected error posting tweet: {str(e)}")
                return None

    # Replace the original post_tweet with the enhanced version
    global post_tweet
    post_tweet = enhanced_post_tweet

    logger.info("Future improvements implemented.")

def display_engagement_summary():
    engagement_data = load_engagement_data()
    for tweet_id, data_points in engagement_data.items():
        print(f"Tweet {tweet_id}:")
        for point in data_points:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(point['timestamp']))
            engagement = point['engagement']
            print(f"  {timestamp}: Retweets: {engagement['retweets']}, Likes: {engagement['likes']}, Replies: {engagement['replies']}")
        print()
    logger.info("Engagement summary displayed")

def display_sentiment_summary():
    sentiment_data = load_sentiment_data()
    print("Sentiment Analysis Summary:")
    for tweet_id, data in sentiment_data.items():
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['timestamp']))
        print(f"Tweet {tweet_id}:")
        print(f"  Posted at: {timestamp}")
        print(f"  Content: {data['content']}")
        print(f"  Sentiment: {data['sentiment']}")
        print()
    logger.info("Sentiment summary displayed")

def initialize_post_cap_tracker(monthly_cap=500000):
    global post_cap_tracker
    post_cap_tracker = PostCapTracker(monthly_cap=monthly_cap)

def safe_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        return None

def get_random_image(image_folder):
    logger.info(f"Attempting to get random image from {image_folder}")
    try:
        if not os.path.exists(image_folder):
            logger.warning(f"Image folder does not exist: {image_folder}")
            return None
        
        images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if images:
            selected_image = random.choice(images)
            logger.info(f"Random image selected: {selected_image}")
            return os.path.join(image_folder, selected_image)
        else:
            logger.warning(f"No images found in {image_folder}")
            return None
    except Exception as e:
        logger.exception(f"Error getting random image: {str(e)}")
        return None

def upload_media(image_path):
    try:
        media = api.media_upload(filename=image_path)
        logger.info(f"Media uploaded successfully: {image_path}")
        return media
    except Exception as e:
        logger.error(f"Error uploading media: {str(e)}")
        return None

def validate_tweet_content(content):
    # Check length
    if len(content) > 280:
        return False, "Tweet exceeds 280 characters"
    
    # Check for prohibited content (example)
    prohibited_words = ["spam", "abuse", "offensive"]
    if any(word in content.lower() for word in prohibited_words):
        return False, "Tweet contains prohibited content"
    
    return True, "Content is valid"

COOLDOWN_PERIOD = 15 * 60  # 15 minutes

def post_tweet_with_cooldown(content):
    try:
        return client.create_tweet(text=content)
    except tweepy.errors.Forbidden as e:
        logger.error(f"403 Forbidden error: {str(e)}")
        logger.info(f"Entering cooldown period of {COOLDOWN_PERIOD} seconds")
        time.sleep(COOLDOWN_PERIOD)
        return None

def main():
    global client, api, post_cap_tracker
    
    logger.info("Twitter bot started")
    
    config = load_config()
    client, api = initialize_twitter_client(
        config['TWITTER_API_KEY'],
        config['TWITTER_API_SECRET'],
        config['TWITTER_ACCESS_TOKEN'],
        config['TWITTER_ACCESS_TOKEN_SECRET']
    )
    initialize_openai(config['OPENAI_API_KEY'])
    
    initialize_post_cap_tracker()
    
    implement_future_improvements()
    
    # Post an immediate tweet
    logger.info("Posting initial tweet")
    initial_tweet_posted = False
    while not initial_tweet_posted:
        initial_tweet_posted = job()
        if not initial_tweet_posted:
            logger.warning("Failed to post initial tweet. Retrying in 60 seconds.")
            time.sleep(60)
    logger.info("Initial tweet posted successfully")
    
    schedule.every().hour.at(":00").do(job)
    schedule_random_job()
    
    # Schedule daily summaries
    schedule.every().day.at("00:00").do(display_engagement_summary)
    schedule.every().day.at("00:05").do(display_sentiment_summary)

    logger.info("Entering main loop")
    while True:
        try:
            schedule.run_pending()
            
            if not schedule.get_jobs('random_job'):
                schedule_random_job()
            
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying

    logger.info("Main loop exited")  # This line should never be reached under normal circumstances

if __name__ == "__main__":
    print("Script started")
    logger.info("Script started")
    logger.info("Starting the Twitter bot")
    try:
        main()
    except KeyboardInterrupt:
        print("Bot stopped by user")
        logger.info("Bot stopped by user")
    except Exception as e:
        print(f"An error occurred while running the bot: {str(e)}")
        logger.exception(f"An error occurred while running the bot: {str(e)}")
    finally:
        print("Bot shutting down")
        logger.info("Bot shutting down")

    logger.info("Environment variables after loading .env:")
    for key in ['TWITTER_API_KEY', 'TWITTER_API_SECRET', 'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_TOKEN_SECRET', 'OPENAI_API_KEY']:
        logger.info(f"{key}: {'Set' if os.getenv(key) else 'Not set'}")

    print("Environment variables:")
    for key in ['TWITTER_API_KEY', 'TWITTER_API_SECRET', 'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_TOKEN_SECRET', 'OPENAI_API_KEY']:
        print(f"{key}: {'Set' if os.getenv(key) else 'Not set'}")

print("Script ended")
