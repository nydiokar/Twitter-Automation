import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
import openai
from textblob import TextBlob
import time
import datetime
import tweepy

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from twitter_automation_bot.src import tweet_bot

class TestTwitterBot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_client = MagicMock()
        cls.mock_env = {
            'TWITTER_API_KEY': 'mock_api_key',
            'TWITTER_API_SECRET': 'mock_api_secret',
            'TWITTER_ACCESS_TOKEN': 'mock_access_token',
            'TWITTER_ACCESS_TOKEN_SECRET': 'mock_access_token_secret',
            'OPENAI_API_KEY': 'mock_openai_key'
        }
        
        # Mock the load_config function
        cls.patcher = patch('twitter_automation_bot.src.tweet_bot.load_config', return_value=cls.mock_env)
        cls.patcher.start()
        
        # Initialize the client and other global variables
        tweet_bot.client = cls.mock_client
        tweet_bot.initialize_post_cap_tracker(monthly_cap=500000)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_initialize_twitter_client(self):
        with patch('tweepy.Client') as mock_tweepy_client:
            tweet_bot.initialize_twitter_client(
                self.mock_env['TWITTER_API_KEY'],
                self.mock_env['TWITTER_API_SECRET'],
                self.mock_env['TWITTER_ACCESS_TOKEN'],
                self.mock_env['TWITTER_ACCESS_TOKEN_SECRET']
            )
            mock_tweepy_client.assert_called_once()

    def test_post_tweet(self):
        mock_response = MagicMock()
        mock_response.data = {'id': '67890', 'text': 'Test tweet'}
        self.mock_client.create_tweet.return_value = mock_response

        with patch('twitter_automation_bot.src.tweet_bot.save_posted_tweet') as mock_save:
            tweet_id = tweet_bot.post_tweet('Test tweet')
            self.assertEqual(tweet_id, '67890')
            mock_save.assert_called_once_with('Test tweet')

    def test_track_engagement(self):
        mock_tweet = MagicMock()
        mock_tweet.data.public_metrics = {
            'retweet_count': 5,
            'reply_count': 2,
            'like_count': 10,
            'quote_count': 1
        }
        self.mock_client.get_tweet.return_value = mock_tweet

        with patch('twitter_automation_bot.src.tweet_bot.save_engagement_data') as mock_save:
            engagement = tweet_bot.track_engagement('67890')
            self.assertEqual(engagement['retweets'], 5)
            self.assertEqual(engagement['replies'], 2)
            self.assertEqual(engagement['likes'], 10)
            mock_save.assert_called_once()

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('json.dump')
    @patch('twitter_automation_bot.src.tweet_bot.load_posted_tweets', return_value=[])
    def test_save_posted_tweet(self, mock_load, mock_json_dump, mock_open):
        tweet_data = 'Test tweet'
        tweet_bot.save_posted_tweet(tweet_data)
        mock_open.assert_called_once_with(tweet_bot.POSTED_TWEETS_FILE, 'w')
        mock_json_dump.assert_called_once()

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('json.dump')
    @patch('twitter_automation_bot.src.tweet_bot.load_engagement_data', return_value={})
    def test_save_engagement_data(self, mock_load, mock_json_dump, mock_open):
        engagement_data = {'retweets': 5, 'replies': 2, 'likes': 10, 'quotes': 1}
        tweet_bot.save_engagement_data('67890', engagement_data)
        mock_open.assert_called_once_with(tweet_bot.ENGAGEMENT_FILE, 'w')
        mock_json_dump.assert_called_once()

    @patch('openai.ChatCompletion.create')
    def test_generate_content(self, mock_openai_create):
        mock_response = MagicMock()
        mock_response.choices = [{'message': {'content': 'Generated tweet content'}}]
        mock_openai_create.return_value = mock_response

        content = tweet_bot.generate_content('Test prompt')
        self.assertEqual(content, 'Generated tweet content')
        mock_openai_create.assert_called_once()

    @patch('twitter_automation_bot.src.tweet_bot.generate_content')
    def test_get_unique_content(self, mock_generate_content):
        mock_generate_content.return_value = 'Unique tweet content'
        with patch('twitter_automation_bot.src.tweet_bot.load_posted_tweets', return_value=[]):
            content = tweet_bot.get_unique_content('Test prompt')
            self.assertEqual(content, 'Unique tweet content')
            mock_generate_content.assert_called_once_with('Test prompt')

    @patch('twitter_automation_bot.src.tweet_bot.get_unique_content')
    def test_job(self, mock_get_unique_content):
        mock_get_unique_content.return_value = 'Generated tweet'
        mock_response = MagicMock()
        mock_response.data = {'id': '12345'}
        self.mock_client.create_tweet.return_value = mock_response

        with patch('twitter_automation_bot.src.tweet_bot.post_tweet') as mock_post_tweet, \
             patch('twitter_automation_bot.src.tweet_bot.track_engagement') as mock_track_engagement, \
             patch('twitter_automation_bot.src.tweet_bot.analyze_sentiment') as mock_analyze_sentiment, \
             patch('twitter_automation_bot.src.tweet_bot.load_sentiment_data', return_value={}), \
             patch('twitter_automation_bot.src.tweet_bot.save_sentiment_data') as mock_save_sentiment:

            mock_post_tweet.return_value = '12345'
            mock_analyze_sentiment.return_value = 'positive'

            tweet_bot.job()

            mock_get_unique_content.assert_called_once()
            mock_post_tweet.assert_called_once_with('Generated tweet')
            mock_track_engagement.assert_called_once_with('12345')
            mock_analyze_sentiment.assert_called_once_with('Generated tweet')
            mock_save_sentiment.assert_called_once()

    def test_custom_tweet_retrieval(self):
        # Implement test for custom tweet retrieval
        pass

    def test_sentiment_analysis(self):
        # Implement test for sentiment analysis
        pass

    def test_rate_limiting(self):
        # Implement test for rate limiting and backoff strategies
        pass

    def test_job_scheduling(self):
        # Implement test for job scheduling
        pass

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{}')
    @patch('json.load')
    def test_load_engagement_data(self, mock_json_load, mock_open, mock_exists):
        mock_json_load.return_value = {}  # Assuming the file is initially empty
        result = tweet_bot.load_engagement_data()
        self.assertEqual(result, {})
        mock_open.assert_called_once_with(tweet_bot.ENGAGEMENT_FILE, 'r')
        mock_json_load.assert_called_once()

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='[]')
    @patch('json.load')
    def test_load_posted_tweets(self, mock_json_load, mock_open):
        mock_json_load.return_value = ['Tweet 1', 'Tweet 2']
        result = tweet_bot.load_posted_tweets()
        self.assertEqual(result, ['Tweet 1', 'Tweet 2'])
        mock_open.assert_called_once_with(tweet_bot.POSTED_TWEETS_FILE, 'r')
        mock_json_load.assert_called_once()

class TestRateLimiter(unittest.TestCase):
    def test_rate_limiter(self):
        limiter = tweet_bot.RateLimiter(max_calls=2, period=1)
        
        @limiter
        def test_func():
            return True

        # First two calls should be immediate
        start = time.time()
        self.assertTrue(test_func())
        self.assertTrue(test_func())
        self.assertLess(time.time() - start, 0.1)

        # Third call should be delayed
        start = time.time()
        self.assertTrue(test_func())
        self.assertGreater(time.time() - start, 0.9)

class TestPostCapTracker(unittest.TestCase):
    def test_post_cap_tracker(self):
        tracker = tweet_bot.PostCapTracker(monthly_cap=5)
        
        for _ in range(5):
            self.assertTrue(tracker.can_make_request())
            tracker.increment_count()
        
        self.assertFalse(tracker.can_make_request())

        # Simulate month change
        tracker.current_month = (datetime.datetime.now().month - 1) % 12
        self.assertTrue(tracker.can_make_request())

class TestSentimentAnalysis(unittest.TestCase):
    def test_analyze_sentiment(self):
        self.assertEqual(tweet_bot.analyze_sentiment("I love this!"), "positive")
        self.assertEqual(tweet_bot.analyze_sentiment("I hate this."), "negative")
        self.assertEqual(tweet_bot.analyze_sentiment("The sky is blue."), "neutral")

class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        # Ensure post_cap_tracker is initialized for each test
        tweet_bot.initialize_post_cap_tracker(monthly_cap=500000)

    @patch('tweepy.Client.create_tweet')
    def test_post_tweet_error_handling(self, mock_create_tweet):
        # Use tweepy.errors.TweepError if available, otherwise use Exception
        try:
            from tweepy.errors import TweepError
        except ImportError:
            TweepError = Exception
        
        mock_create_tweet.side_effect = TweepError("Rate limit exceeded")
        
        with self.assertLogs(level='ERROR') as log:
            result = tweet_bot.post_tweet("Test tweet")
            self.assertIsNone(result)
            self.assertIn("Error posting tweet", log.output[0])

class TestIntegration(unittest.TestCase):
    @patch('twitter_automation_bot.src.tweet_bot.generate_content')
    @patch('twitter_automation_bot.src.tweet_bot.post_tweet')
    @patch('twitter_automation_bot.src.tweet_bot.track_engagement')
    @patch('twitter_automation_bot.src.tweet_bot.analyze_sentiment')
    @patch('twitter_automation_bot.src.tweet_bot.load_sentiment_data')
    @patch('twitter_automation_bot.src.tweet_bot.save_sentiment_data')
    def test_full_cycle(self, mock_save_sentiment, mock_load_sentiment, mock_analyze, mock_track, mock_post, mock_generate):
        mock_generate.return_value = "Test tweet"
        mock_post.return_value = "12345"
        mock_track.return_value = {"likes": 10, "retweets": 5}
        mock_analyze.return_value = "positive"
        mock_load_sentiment.return_value = {}

        tweet_bot.job()

        mock_generate.assert_called_once()
        mock_post.assert_called_once_with("Test tweet")
        mock_track.assert_called_once_with("12345")
        mock_analyze.assert_called_once_with("Test tweet")
        mock_load_sentiment.assert_called_once()
        mock_save_sentiment.assert_called_once()

class TestParameterized(unittest.TestCase):
    @patch('twitter_automation_bot.src.tweet_bot.is_similar')
    @patch('twitter_automation_bot.src.tweet_bot.generate_content')
    def test_get_unique_content_scenarios(self, mock_generate, mock_is_similar):
        scenarios = [
            (["Tweet 1"], False, "Unique tweet"),
            (["Tweet 1"], True, None),
        ]

        for posted_tweets, is_similar, expected_result in scenarios:
            with self.subTest(posted_tweets=posted_tweets, is_similar=is_similar):
                mock_generate.return_value = "Unique tweet"
                mock_is_similar.return_value = is_similar

                with patch('twitter_automation_bot.src.tweet_bot.load_posted_tweets', return_value=posted_tweets):
                    result = tweet_bot.get_unique_content("Test prompt")
                    self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
