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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollegeBasketballPredictor:
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

    def get_todays_games(self) -> List[Dict]:
        """Fetch all college basketball games scheduled for today from ESPN API."""
        try:
            url = f"{self.base_url}/scoreboard?dates=20250218"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            print(data.get("events"))
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
                except (KeyError, IndexError, ValueError):
                    over_under = None
                
                game = {
                    'home_team': home_team['team']['displayName'],
                    'away_team': away_team['team']['displayName'],
                    'home_id': home_team['team']['id'],
                    'away_id': away_team['team']['id'],
                    'over_under': over_under
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
                'avg_total_score': (total_points + total_points_allowed) / total_games if total_games > 0 else 0
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
                url = f"{self.base_url}/scoreboard?dates={date}"
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

    def prepare_training_data(self) -> pd.DataFrame:
        """Prepare historical data for model training."""
        training_data = []
        
        # Get today's games to get team IDs
        games = self.get_todays_games()
        
        # Get historical data for each team in today's games
        for game in games:
            home_stats = self.get_team_stats(game['home_id'], self.lookback_days)
            away_stats = self.get_team_stats(game['away_id'], self.lookback_days)
            
            if home_stats and away_stats:
                feature_row = {
                    'home_win_pct': home_stats['win_percentage'],
                    'home_avg_points': home_stats['avg_points'],
                    'home_avg_points_allowed': home_stats['avg_points_allowed'],
                    'away_win_pct': away_stats['win_percentage'],
                    'away_avg_points': away_stats['avg_points'],
                    'away_avg_points_allowed': away_stats['avg_points_allowed'],
                    'total_score': home_stats['avg_total_score']  # Using historical total as target
                }
                print(feature_row)
                training_data.append(feature_row)
        
        return pd.DataFrame(training_data)

    def predict_today_scores(self):
        """Main function to predict scores for today's games."""
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.ensemble import GradientBoostingRegressor
        from scipy import stats
        
        games = self.get_todays_games()
        if not games:
            logger.info("No games found for today")
            return

        training_data = self.prepare_training_data()
        if training_data.empty:
            logger.error("No training data available")
            return

        # Define feature columns
        feature_columns = [
            'home_win_pct',
            'home_avg_points',
            'home_avg_points_allowed',
            'away_win_pct',
            'away_avg_points',
            'away_avg_points_allowed'
        ]

        # Prepare features and target
        X = training_data[feature_columns]
        y = training_data['total_score']

        # Initialize models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        # Determine number of splits based on data size
        n_splits = min(5, len(X) - 1)
        if n_splits < 2:
            logger.warning("Not enough data for cross-validation. Using single train-test split.")
            rmse_scores = np.array([0.0])
        else:
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(rf_model, X, y, cv=kfold, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-cv_scores)

        # Train both models on all data
        rf_model.fit(X, y)
        gb_model.fit(X, y)

        # Make predictions for today's games
        for game in games:
            home_stats = self.get_team_stats(game['home_id'], self.lookback_days)
            away_stats = self.get_team_stats(game['away_id'], self.lookback_days)
            
            if not home_stats or not away_stats:
                continue

            # Create features array without column names for tree predictions
            features_array = np.array([[
                home_stats['win_percentage'],
                home_stats['avg_points'],
                home_stats['avg_points_allowed'],
                away_stats['win_percentage'],
                away_stats['avg_points'],
                away_stats['avg_points_allowed']
            ]])

            # Create DataFrame with column names for main predictions
            features_df = pd.DataFrame(features_array, columns=feature_columns)

            # Get predictions from both models using DataFrame
            rf_pred = rf_model.predict(features_df)[0]
            gb_pred = gb_model.predict(features_df)[0]
            
            # Get individual tree predictions using numpy array
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
            
            result = (
                f"\n{game['home_team']} vs {game['away_team']}\n"
                f"Predicted Total: {mean_prediction:.1f} ± {confidence_interval:.1f}\n"
                f"Vegas Total: {game['over_under'] if game['over_under'] else 'N/A'}\n"
                f"Model Confidence: {confidence_score:.1f}%\n"
                f"Model Agreement: {model_agreement:.1f}%\n"
                f"Prediction Range: {mean_prediction - confidence_interval:.1f} to {mean_prediction + confidence_interval:.1f}\n"
            )
            
            if game['over_under']:
                result += (
                    f"Over/Under Confidence: {over_under_confidence}\n"
                    f"Recommendation: {strength} "
                    f"{'OVER' if mean_prediction > game['over_under'] else 'UNDER'}\n"
                )
            
            if n_splits >= 2:
                result += f"Model RMSE: {rmse_scores.mean():.1f}\n"
            else:
                result += "Warning: Limited data available for model validation\n"
            
            logger.info(result)

def main():
    predictor = CollegeBasketballPredictor(lookback_days=30)
    predictor.predict_today_scores()

if __name__ == "__main__":
    main()
