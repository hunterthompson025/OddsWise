import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from typing import List, Dict
import logging
import time
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollegeBasketballPredictor:
    def __init__(self, lookback_days: int = 30, prediction_type: str = 'total'):
        self.lookback_days = lookback_days
        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ['total', 'spread']:
            raise ValueError("prediction_type must be either 'total' or 'spread'")
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

    def get_todays_games(self) -> List[Dict]:
        """Fetch all college basketball games scheduled for today from ESPN API."""
        try:
            today = datetime.now().strftime('%Y%m%d')
            url = f"{self.base_url}/scoreboard?dates={today}&groups=50"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            games = []
            for event in data.get("events", []):
                competition = event['competitions'][0]
                
                # Skip completed or in-progress games
                status = competition['status']['type']
                if status['completed'] or status['name'] in ['STATUS_IN_PROGRESS', 'STATUS_HALFTIME']:
                    continue
                
                home_team = competition['competitors'][0]
                away_team = competition['competitors'][1]
                try:
                    over_under = float(competition['odds'][0]['overUnder'])
                    spread = float(competition['odds'][0]['spread'])
                except (KeyError, IndexError, ValueError):
                    over_under = None
                    spread = None
                
                game = {
                    'home_team': home_team['team']['displayName'],
                    'away_team': away_team['team']['displayName'],
                    'home_id': home_team['team']['id'],
                    'away_id': away_team['team']['id'],
                    'over_under': over_under,
                    'spread': spread
                }
                games.append(game)
                
            return games
            
        except Exception as e:
            logger.error(f"Error fetching today's games: {str(e)}")
            return []

    def get_team_stats(self, team_id: str, days_back: int) -> Dict:
        """Get team statistics from historical games."""
        try:
            historical_games = self.get_historical_games(team_id, days_back)
            
            if not historical_games:
                return {}
            
            total_games = len(historical_games)
            wins = sum(1 for game in historical_games if game['is_winner'])
            total_points = sum(game['points'] for game in historical_games)
            total_points_allowed = sum(game['opponent_points'] for game in historical_games)
            
            stats = {
                'win_percentage': wins / total_games if total_games > 0 else 0,
                'avg_points': total_points / total_games if total_games > 0 else 0,
                'avg_points_allowed': total_points_allowed / total_games if total_games > 0 else 0,
                'avg_total_score': (total_points + total_points_allowed) / total_games if total_games > 0 else 0,
                'avg_margin': (total_points - total_points_allowed) / total_games if total_games > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error fetching team stats: {str(e)}")
            return {}

    def get_historical_games(self, team_id: str, days_back: int) -> List[Dict]:
        """Get historical game results using the ESPN API."""
        games = []
        
        for day_offset in range(days_back):
            date = (datetime.now() - timedelta(days=day_offset)).strftime('%Y%m%d')
            try:
                url = f"{self.base_url}/scoreboard?dates={date}&groups=50"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                
                for event in data.get('events', []):
                    competition = event['competitions'][0]
                    
                    # Only include completed games
                    if not competition['status']['type']['completed']:
                        continue
                    
                    home_team = competition['competitors'][0]
                    away_team = competition['competitors'][1]
                    
                    # Check if this game involves the team we're looking for
                    if team_id in [home_team['team']['id'], away_team['team']['id']]:
                        team_data = home_team if home_team['team']['id'] == team_id else away_team
                        opponent_data = away_team if home_team['team']['id'] == team_id else home_team
                        
                        try:
                            game = {
                                'date': date,
                                'points': int(team_data['score']),
                                'opponent_points': int(opponent_data['score']),
                                'is_winner': team_data.get('winner', False),
                                'is_home': team_data['homeAway'] == 'home'
                            }
                            games.append(game)
                        except (KeyError, ValueError):
                            continue
                
                time.sleep(0.1)  # Small delay to avoid rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching historical games for date {date}: {str(e)}")
                continue
                
        return games

    def predict_game(self, game: Dict) -> Dict:
        """Predict the outcome for a specific game."""
        home_stats = self.get_team_stats(game['home_id'], self.lookback_days)
        away_stats = self.get_team_stats(game['away_id'], self.lookback_days)
        
        if not home_stats or not away_stats:
            return None

        # Create features array
        features = [
            home_stats['win_percentage'],
            home_stats['avg_points'],
            home_stats['avg_points_allowed'],
            away_stats['win_percentage'],
            away_stats['avg_points'],
            away_stats['avg_points_allowed'],
        ]

        # Initialize models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        # Create training data
        training_data = []
        for _ in range(100):  # Generate synthetic training data
            training_data.append({
                'home_win_pct': np.random.random(),
                'home_avg_points': np.random.normal(70, 10),
                'home_avg_points_allowed': np.random.normal(70, 10),
                'away_win_pct': np.random.random(),
                'away_avg_points': np.random.normal(70, 10),
                'away_avg_points_allowed': np.random.normal(70, 10),
                'target': np.random.normal(140, 15)  # Total points
            })
        
        training_df = pd.DataFrame(training_data)
        X = training_df.drop('target', axis=1)
        y = training_df['target']

        # Train models
        rf_model.fit(X, y)
        gb_model.fit(X, y)

        # Make predictions
        features_array = np.array([features])
        features_df = pd.DataFrame(features_array, columns=X.columns)
        
        rf_pred = rf_model.predict(features_df)[0]
        gb_pred = gb_model.predict(features_df)[0]
        
        # Get individual tree predictions
        tree_predictions = np.array([tree.predict(features_array)[0] 
                                   for tree in rf_model.estimators_])
        
        # Calculate prediction statistics
        mean_prediction = np.mean([rf_pred, gb_pred])
        std_prediction = np.std(tree_predictions)
        confidence_interval = std_prediction * 1.96  # 95% confidence interval
        
        # Calculate model agreement score (0-100%)
        model_agreement = 100 * (1 - abs(rf_pred - gb_pred) / mean_prediction)
        
        # Calculate prediction confidence score (0-100%)
        confidence_score = min(100, max(0, 100 * (1 - (std_prediction / mean_prediction))))
        
        # Calculate over/under confidence if Vegas line exists
        over_under_confidence = "N/A"
        recommendation = "N/A"
        if game['over_under']:
            # Calculate z-score for Vegas line
            z_score = abs(mean_prediction - game['over_under']) / std_prediction
            # Convert to probability
            prob = stats.norm.cdf(z_score)
            over_under_confidence = f"{prob * 100:.1f}%"
            
            # Determine recommendation strength
            if prob > 0.8:
                strength = "Strong"
            elif prob > 0.65:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            recommendation = f"{strength} {'OVER' if mean_prediction > game['over_under'] else 'UNDER'}"

        return {
            'predicted_total': mean_prediction,
            'confidence_interval': confidence_interval,
            'model_confidence': confidence_score,
            'model_agreement': model_agreement,
            'prediction_low': mean_prediction - confidence_interval,
            'prediction_high': mean_prediction + confidence_interval,
            'over_under_confidence': over_under_confidence,
            'recommendation': recommendation
        }

def main():
    st.title("College Basketball Game Predictor")
    
    # Initialize predictor
    predictor = CollegeBasketballPredictor(lookback_days=30, prediction_type='total')
    
    # Get today's games
    games = predictor.get_todays_games()
    
    if not games:
        st.error("No games found for today")
        return
    
    # Create dropdown options
    game_options = [f"{game['home_team']} vs {game['away_team']}" for game in games]
    
    # Add dropdown
    selected_game = st.selectbox("Select a game:", game_options)
    
    # Get the selected game data
    selected_game_data = games[game_options.index(selected_game)]
    
    # Display game details
    st.subheader("Game Details")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Home Team:** {selected_game_data['home_team']}")
    with col2:
        st.write(f"**Away Team:** {selected_game_data['away_team']}")
    
    if selected_game_data['over_under']:
        st.write(f"**Vegas Over/Under:** {selected_game_data['over_under']}")
    
    # Display last 5 games for each team
    st.subheader("Recent Games")
    
    # Get historical games for both teams
    home_games = predictor.get_historical_games(selected_game_data['home_id'], 30)[:5]  # Last 5 games
    away_games = predictor.get_historical_games(selected_game_data['away_id'], 30)[:5]  # Last 5 games
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{selected_game_data['home_team']} - Last 5 Games**")
        for game in home_games:
            result = "W" if game['is_winner'] else "L"
            location = "Home" if game['is_home'] else "Away"
            total_points = game['points'] + game['opponent_points']
            formatted_date = datetime.strptime(game['date'], '%Y%m%d').strftime('%m/%d/%Y')
            st.write(f"{formatted_date}: {result} ({location}) - {game['points']}-{game['opponent_points']} (Total: {total_points})")
    
    with col2:
        st.write(f"**{selected_game_data['away_team']} - Last 5 Games**")
        for game in away_games:
            result = "W" if game['is_winner'] else "L"
            location = "Home" if game['is_home'] else "Away"
            total_points = game['points'] + game['opponent_points']
            formatted_date = datetime.strptime(game['date'], '%Y%m%d').strftime('%m/%d/%Y')
            st.write(f"{formatted_date}: {result} ({location}) - {game['points']}-{game['opponent_points']} (Total: {total_points})")

    # Get prediction
    prediction = predictor.predict_game(selected_game_data)
    
    if prediction:
        st.subheader("Prediction Results")
        
        # Display prediction metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Total", f"{prediction['predicted_total']:.1f}")
            st.metric("Model Confidence", f"{prediction['model_confidence']:.1f}%")
            st.metric("Model Agreement", f"{prediction['model_agreement']:.1f}%")
        
        with col2:
            st.metric("Confidence Interval", f"±{prediction['confidence_interval']:.1f}")
            st.metric("Prediction Range", f"{prediction['prediction_low']:.1f} - {prediction['prediction_high']:.1f}")
            st.metric("Over/Under Confidence", prediction['over_under_confidence'])
        
        # Display recommendation
        st.subheader("Recommendation")
        st.write(f"**{prediction['recommendation']}**")
        
        # Add a visual representation of the prediction
        st.subheader("Prediction Visualization")
        if selected_game_data['over_under']:
            # Create figure and axis objects with matplotlib
            fig, ax = plt.subplots()
            data = pd.DataFrame({
                'Value': [prediction['predicted_total'], selected_game_data['over_under']],
                'Type': ['Predicted Total', 'Vegas Line']
            })
            data.plot(kind='bar', ax=ax)
            plt.close()  # Close the figure to prevent display in non-Streamlit contexts
            
            # Display the plot in Streamlit
            st.pyplot(fig)
    else:
        st.error("Unable to generate prediction for this game")

if __name__ == "__main__":
    main() 