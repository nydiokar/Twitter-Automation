# Twitter Automation Bot

This project is a Twitter automation bot that generates and posts tweets about whatever you want.

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
   git clone https://github.com/nydiokar/twitter-automation-bot.git
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

**Before running the script:**

1. Ensure you have /images folder and it's filled in with desired images to later post. Pick random image every time.

2. Include desired prompts to #List of prompts because it is using prompts from this list every time it reaches out to the model selected.

3. custom_tweets.txt file is always checked with priority and if there is content there will always post it first. The posted content will be deleted afterwards.

4. In generate_content() change:
 - "temperature" for variability; 
- "model=" replace the model name with the one you are using to call and generate the content; 
- {"role": "system", "content": **THE SYSTEM PROMPT YOU ARE CALLING TO THE MODEL**}, change this system prompt to what suites the best your project.  

Run this script with:

```
python twitter_automation_bot/src/tweet_bot.py
```
## Project Structure

- `twitter_automation_bot/`: Contains the source code for the bot.
- `logs/`: Directory for log files.
- `data/`: Directory for data files.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
