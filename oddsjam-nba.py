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
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBAPredictor:
    def __init__(self, lookback_days: int = 30, prediction_type: str = 'total', 
                 spreadsheet_id: str = None, credentials_path: str = None):
        self.lookback_days = lookback_days
        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ['total', 'spread']:
            raise ValueError("prediction_type must be either 'total' or 'spread'")
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"  # Changed to NBA endpoint
        self.spreadsheet_id = spreadsheet_id
        self.credentials_path = credentials_path

    def append_to_google_sheet(self, values):
        """Append data to Google Sheet."""
        try:
            if not self.spreadsheet_id or not self.credentials_path:
                logger.warning("Google Sheets credentials not provided. Skipping sheet update.")
                return

            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            # Disable cache discovery to prevent warning
            service = build('sheets', 'v4', credentials=credentials, cache_discovery=False)
            
            today = 'NBA_' + datetime.now().strftime('%Y-%m-%d')
            
            try:
                service.spreadsheets().get(
                    spreadsheetId=self.spreadsheet_id,
                    ranges=today
                ).execute()
            except HttpError:
                body = {
                    'requests': [{
                        'addSheet': {
                            'properties': {
                                'title': today
                            }
                        }
                    }]
                }
                service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body=body
                ).execute()

            body = {
                'values': values
            }
            service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=f'{today}!A1',
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            logger.info(f"Successfully appended data to Google Sheet")
            
        except Exception as e:
            logger.error(f"Error appending to Google Sheet: {str(e)}")

    def get_todays_games(self) -> List[Dict]:
        """Fetch all NBA games scheduled for today from ESPN API."""
        try:
            url = f"{self.base_url}/scoreboard?dates=20250224"  # NBA scoreboard endpoint
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
                print('Scoring ', home_team['team']['displayName'], 'vs', away_team['team']['displayName'])
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
                url = f"{self.base_url}/scoreboard?dates={date}"  # NBA scoreboard with date
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
        
        games = self.get_todays_games()
        
        for game in games:
            home_stats = self.get_team_stats(game['home_id'], self.lookback_days)
            away_stats = self.get_team_stats(game['away_id'], self.lookback_days)
            
            if home_stats and away_stats:
                feature_row = {
                    'home_win_pct': home_stats['win_percentage'],
                    'home_avg_points': home_stats['avg_points'],
                    'home_avg_points_allowed': home_stats['avg_points_allowed'],
                    'home_avg_margin': home_stats['avg_margin'],
                    'away_win_pct': away_stats['win_percentage'],
                    'away_avg_points': away_stats['avg_points'],
                    'away_avg_points_allowed': away_stats['avg_points_allowed'],
                    'away_avg_margin': away_stats['avg_margin'],
                }

                if self.prediction_type == 'total':
                    feature_row['target'] = home_stats['avg_total_score']
                else:  # spread
                    feature_row['target'] = home_stats['avg_margin'] - away_stats['avg_margin']

                training_data.append(feature_row)
        
        return pd.DataFrame(training_data)

    def predict_today_scores(self):
        """Main function to predict scores for today's games."""
        games = self.get_todays_games()
        if not games:
            logger.info("No games found for today")
            return

        training_data = self.prepare_training_data()
        if training_data.empty:
            logger.error("No training data available")
            return

        # Define feature columns based on prediction type
        feature_columns = [
            'home_win_pct',
            'home_avg_points',
            'home_avg_points_allowed',
            'away_win_pct',
            'away_avg_points',
            'away_avg_points_allowed',
        ]
        
        if self.prediction_type == 'spread':
            feature_columns.extend(['home_avg_margin', 'away_avg_margin'])

        # Prepare features and target
        X = training_data[feature_columns]
        y = training_data['target']

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

        # Store all rows for batch upload
        sheet_rows = []
        
        for game in games:
            home_stats = self.get_team_stats(game['home_id'], self.lookback_days)
            away_stats = self.get_team_stats(game['away_id'], self.lookback_days)
            
            if not home_stats or not away_stats:
                continue

            # Create features array based on prediction type
            features = [
                home_stats['win_percentage'],
                home_stats['avg_points'],
                home_stats['avg_points_allowed'],
                away_stats['win_percentage'],
                away_stats['avg_points'],
                away_stats['avg_points_allowed'],
            ]
            
            if self.prediction_type == 'spread':
                features.extend([home_stats['avg_margin'], away_stats['avg_margin']])

            features_array = np.array([features])
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
            spread_confidence = "N/A"
            if game['spread'] and self.prediction_type == 'spread':
                # Calculate z-score for Vegas spread
                z_score = abs(mean_prediction - game['spread']) / std_prediction
                # Convert to probability
                prob = stats.norm.cdf(z_score)
                spread_confidence = f"{prob * 100:.1f}%"
                
                # Determine recommendation strength
                if prob > 0.8:
                    strength = "Strong"
                elif prob > 0.65:
                    strength = "Moderate"
                else:
                    strength = "Weak"
            elif game['over_under'] and self.prediction_type == 'total':
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
            
            # Adjust confidence calculations and output based on prediction type
            if self.prediction_type == 'total':
                line = game['over_under']
                if games.index(game) == 0:
                    header = "Home Team,Away Team,Predicted Total,Confidence Interval,Vegas Total,Model Confidence,Model Agreement,Prediction Low,Prediction High,Over/Under Confidence,Recommendation,RMSE"
                if line:
                    recommendation = f"{strength} {'OVER' if mean_prediction > line else 'UNDER'}"
            else:  # spread
                line = game['spread']
                if games.index(game) == 0:
                    header = "Home Team,Away Team,Predicted Result,Confidence Interval,Vegas Spread,Model Confidence,Model Agreement,Prediction Low,Prediction High,Spread Confidence,Recommendation,RMSE"
                
                # Format the spread prediction as a descriptive string
                if mean_prediction > 0:
                    spread_description = f"{game['home_team']} by {abs(mean_prediction):.1f}"
                else:
                    spread_description = f"{game['away_team']} by {abs(mean_prediction):.1f}"
                
                if line:
                    recommendation = f"{strength} {'HOME' if mean_prediction < line else 'AWAY'}"
                else:
                    recommendation = "N/A"

            # Create CSV row
            rmse_value = f"{rmse_scores.mean():.1f}" if n_splits >= 2 else "N/A"
            csv_row = (
                f"{game['home_team']},{game['away_team']},"
                f"{spread_description if self.prediction_type == 'spread' else f'{mean_prediction:.1f}'},"
                f"{confidence_interval:.1f},"
                f"{line if line else 'N/A'},"
                f"{confidence_score:.1f}%,{model_agreement:.1f}%,"
                f"{mean_prediction - confidence_interval:.1f},"
                f"{mean_prediction + confidence_interval:.1f},"
                f"{over_under_confidence if self.prediction_type == 'total' else spread_confidence},"
                f"{recommendation},"
                f"{rmse_value}"
            )
            
            # Create row data
            row_data = [
                game['home_team'],
                game['away_team'],
                spread_description if self.prediction_type == 'spread' else f'{mean_prediction:.1f}',
                f'{confidence_interval:.1f}',
                str(line if line else 'N/A'),
                f'{confidence_score:.1f}%',
                f'{model_agreement:.1f}%',
                f'{mean_prediction - confidence_interval:.1f}',
                f'{mean_prediction + confidence_interval:.1f}',
                over_under_confidence if self.prediction_type == 'total' else spread_confidence,
                recommendation,
                rmse_value
            ]

            # Add header row if this is the first game
            if games.index(game) == 0:
                header_row = [
                    "Home Team", "Away Team", 
                    "Predicted Result" if self.prediction_type == 'spread' else "Predicted Total",
                    "Confidence Interval", 
                    "Vegas " + ("Spread" if self.prediction_type == 'spread' else "Total"),
                    "Model Confidence", "Model Agreement", "Prediction Low", "Prediction High",
                    "Spread Confidence" if self.prediction_type == 'spread' else "Over/Under Confidence",
                    "Recommendation", "RMSE"
                ]
                sheet_rows.append(header_row)
            
            sheet_rows.append(row_data)
            
            # Also log to console as before
            logger.info(header if games.index(game) == 0 else csv_row)

        # Append all rows to Google Sheet
        self.append_to_google_sheet(sheet_rows)

    def get_completed_games_scores(self) -> List[Dict]:
        """Fetch all completed NBA games from today with their scores."""
        try:
            url = f"{self.base_url}/scoreboard"  # NBA scoreboard endpoint
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            completed_games = []
            
            for event in data.get("events", []):
                competition = event['competitions'][0]
                
                # Only include completed games
                status = competition['status']['type']
                if not status['completed']:
                    continue
                
                home_team = competition['competitors'][0]
                away_team = competition['competitors'][1]
                
                try:
                    home_score = int(home_team['score'])
                    away_score = int(away_team['score'])
                    total_score = home_score + away_score
                    point_spread = home_score - away_score
                    
                    # Try to get the closing lines if available
                    try:
                        closing_total = float(competition['odds'][0]['overUnder'])
                        closing_spread = float(competition['odds'][0]['spread'])
                    except (KeyError, IndexError, ValueError):
                        closing_total = None
                        closing_spread = None
                    
                    game = {
                        'home_team': home_team['team']['displayName'],
                        'away_team': away_team['team']['displayName'],
                        'home_score': home_score,
                        'away_score': away_score,
                        'total_score': total_score,
                        'point_spread': point_spread,
                        'closing_total': closing_total,
                        'closing_spread': closing_spread,
                        'over_under_result': 'OVER' if closing_total and total_score > closing_total else 'UNDER' if closing_total else 'N/A',
                        'spread_result': 'HOME' if closing_spread and point_spread > closing_spread else 'AWAY' if closing_spread else 'N/A'
                    }
                    completed_games.append(game)
                    
                    logger.info(f"Final Score: {home_team['team']['displayName']} {home_score} - {away_team['team']['displayName']} {away_score} "
                              f"(Total: {total_score}, Spread: {point_spread:+})")
                    
                except (KeyError, ValueError) as e:
                    logger.error(f"Error processing game scores: {str(e)}")
                    continue
            
            return completed_games
            
        except Exception as e:
            logger.error(f"Error fetching completed games: {str(e)}")
            return []

def main():
    action = input("What would you like to do? (predict/outcomes): ").lower()
    while action not in ['predict', 'outcomes']:
        action = input("Invalid input. Please enter 'predict' or 'outcomes': ").lower()
    
    # Get Google Sheets configuration
    spreadsheet_id = input("Enter Google Spreadsheet ID (or press Enter to skip): ").strip()
    credentials_path = input("Enter path to service account credentials JSON (or press Enter to skip): ").strip()
    
    if action == 'predict':
        prediction_type = input("Enter prediction type (total/spread): ").lower()
        while prediction_type not in ['total', 'spread']:
            prediction_type = input("Invalid input. Please enter 'total' or 'spread': ").lower()
        
        predictor = NBAPredictor(
            lookback_days=30,
            prediction_type=prediction_type,
            spreadsheet_id=spreadsheet_id if spreadsheet_id else None,
            credentials_path=credentials_path if credentials_path else None
        )
        predictor.predict_today_scores()
    
    else:  # outcomes
        predictor = NBAPredictor(
            lookback_days=30,
            prediction_type='total',  # default value, won't be used for outcomes
            spreadsheet_id=spreadsheet_id if spreadsheet_id else None,
            credentials_path=credentials_path if credentials_path else None
        )
        print("\nCompleted Games Results:")
        completed_games = predictor.get_completed_games_scores()
        
        # If we have Google Sheets configured, append the results
        if predictor.spreadsheet_id and predictor.credentials_path and completed_games:
            results_rows = []
            
            # Add header row
            header_row = [
                "Home Team", "Away Team", "Home Score", "Away Score", 
                "Total Score", "Point Spread", "Closing Total", "Closing Spread",
                "Over/Under Result", "Spread Result"
            ]
            results_rows.append(header_row)
            
            # Add game results
            for game in completed_games:
                row = [
                    game['home_team'],
                    game['away_team'],
                    str(game['home_score']),
                    str(game['away_score']),
                    str(game['total_score']),
                    str(game['point_spread']),
                    str(game['closing_total'] if game['closing_total'] else 'N/A'),
                    str(game['closing_spread'] if game['closing_spread'] else 'N/A'),
                    game['over_under_result'],
                    game['spread_result']
                ]
                results_rows.append(row)
            
            # Append to a new sheet named "Results-YYYY-MM-DD"
            predictor.append_to_google_sheet(results_rows)

if __name__ == "__main__":
    main() 