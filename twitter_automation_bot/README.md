# Twitter Automation Bot

This project is a Twitter automation bot that generates and posts tweets about technology and futurism.

## Features

- Automated tweet generation using AI
- Image attachment to tweets
- Custom tweet scheduling
- Engagement tracking
- Sentiment analysis of posted tweets
- Rate limiting and post cap management

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/twitter-automation-bot.git
   cd twitter-automation-bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables (see Configuration section).

## Configuration

Create a `.env` file in the root directory with the following variables:

TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
OPENAI_API_KEY=your_openai_api_key

Replace the placeholder values with your actual API keys and tokens.

## Usage

Before running the bot:

1. Ensure that the data directory is populated with images. You can add your own images to the data/images directory. 
2. Ensure that the logs directory exists. If it doesn't, the bot will create it.
3. Ensure that the .env file is in the config directory and that the variables are set correctly.

#List of prompts is just examplary, make sure to populate it with prompts by your taste and that the prompts are unique - this is going to prompt the respective model to generate tweets / content and you don't want it to be the same every time - Twitter API call might fail if you post the same tweet twice.

The way you push content by your choice is through the get_custom_tweet() function which will pick a random tweet from the CUSTOM_TWEETS_FILE and remove it from there after posting. The script will every time pick a different tweet from the file, first checking there for content.

Adjust Tempreture parameter in generate_content() function to get different results as well as model = to the kind of model you are calling. For example "gpt-4o" if you want to use GPT-4o model.

Adjust the system message based on your expectations from the model. {"role": "system", "content": "*Your expectations from the model*"}

Engagement function might not work, depending on your API permissions / limits.

Run the bot with:

```
python twitter_automation_bot/src/tweet_bot.py
```
## Project Structure

- `twitter_automation_bot/`: Contains the source code for the bot.
- `logs/`: Directory for log files.
- `data/`: Directory for data files.
- `tests/`: Directory for test files.

## Contributing

We welcome contributions to improve the bot's functionality and performance. Please see our CONTRIBUTING.md for guidelines on how to submit improvements and bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
