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

Run the bot with:

```
python twitter_automation_bot/src/main.py
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
