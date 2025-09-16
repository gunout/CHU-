import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CHUReunionAnalyzer:
    def __init__(self):
        self.chu_nord = "CHU Nord - Saint-Denis"
        self.chu_sud = "CHU Sud - Saint-Pierre"
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', 
                      '#AB83A1', '#5CAB7D', '#2A9D8F', '#E76F51', '#264653']
        
        self.start_year = 2010
        self.end_year = 2024
        
    def generate_health_data(self):
        """Génère des données hospitalières pour les deux CHU de La Réunion"""
        print("🏥 Génération des données hospitalières pour les CHU de La Réunion...")
        
        # Créer une base de données mensuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='M')
        
        data = {'Date': dates}
        
        # Données communes aux deux CHU
        data['Population_Reunion'] = self._simulate_population(dates)
        
        # Données pour CHU Nord
        data['CHU_Nord_Patients'] = self._simulate_nord_patients(dates)
        data['CHU_Nord_Urgences'] = self._simulate_nord_emergencies(dates)
        data['CHU_Nord_Chirurgies'] = self._simulate_nord_surgeries(dates)
        data['CHU_Nord_Naissances'] = self._simulate_nord_births(dates)
        data['CHU_Nord_Budget'] = self._simulate_nord_budget(dates)
        data['CHU_Nord_Personnel'] = self._simulate_nord_staff(dates)
        data['CHU_Nord_Lits'] = self._simulate_nord_beds(dates)
        
        # Données pour CHU Sud
        data['CHU_Sud_Patients'] = self._simulate_sud_patients(dates)
        data['CHU_Sud_Urgences'] = self._simulate_sud_emergencies(dates)
        data['CHU_Sud_Chirurgies'] = self._simulate_sud_surgeries(dates)
        data['CHU_Sud_Naissances'] = self._simulate_sud_births(dates)
        data['CHU_Sud_Budget'] = self._simulate_sud_budget(dates)
        data['CHU_Sud_Personnel'] = self._simulate_sud_staff(dates)
        data['CHU_Sud_Lits'] = self._simulate_sud_beds(dates)
        
        # Données épidémiologiques spécifiques à La Réunion
        data['Dengue_Cas'] = self._simulate_dengue_cases(dates)
        data['Diabete_Cas'] = self._simulate_diabetes_cases(dates)
        data['Cardio_Cas'] = self._simulate_cardio_cases(dates)
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances saisonnières et épidémiologiques
        self._add_health_trends(df)
        
        return df
    
    def _simulate_population(self, dates):
        """Simule la population réunionnaise"""
        base_population = 800000  # population de base en 2010
        
        population = []
        for i, date in enumerate(dates):
            # Croissance démographique annuelle d'environ 0.7%
            growth = 1 + 0.007 * (i / 12)
            population.append(base_population * growth)
        
        return population
    
    def _simulate_nord_patients(self, dates):
        """Simule le nombre de patients pour le CHU Nord"""
        base_patients = 15000  # patients/mois
        
        patients = []
        for i, date in enumerate(dates):
            # Croissance liée à la démographie
            growth = 1 + 0.005 * (i / len(dates))
            
            # Saisonnalité (pic en hiver austral)
            month = date.month
            if month in [6, 7, 8]:  # Hiver austral
                seasonal = 1.15
            else:
                seasonal = 1.0
            
            # Bruit aléatoire
            noise = np.random.normal(1, 0.05)
            
            patients.append(base_patients * growth * seasonal * noise)
        
        return patients
    
    def _simulate_nord_emergencies(self, dates):
        """Simule les passages aux urgences du CHU Nord"""
        base_emergencies = 8000  # urgences/mois
        
        emergencies = []
        for date in dates:
            # Saisonnalité (pic en été avec la dengue)
            month = date.month
            if month in [1, 2, 3]:  # Été austral, saison des pluies
                seasonal = 1.25
            else:
                seasonal = 1.0
            
            noise = np.random.normal(1, 0.08)
            emergencies.append(base_emergencies * seasonal * noise)
        
        return emergencies
    
    def _simulate_nord_surgeries(self, dates):
        """Simule le nombre de chirurgies au CHU Nord"""
        base_surgeries = 1200  # chirurgies/mois
        
        surgeries = []
        for i, date in enumerate(dates):
            # Croissance régulière
            growth = 1 + 0.01 * (i / len(dates))
            noise = np.random.normal(1, 0.06)
            surgeries.append(base_surgeries * growth * noise)
        
        return surgeries
    
    def _simulate_nord_births(self, dates):
        """Simule le nombre de naissances au CHU Nord"""
        base_births = 500  # naissances/mois
        
        births = []
        for date in dates:
            # Légère saisonnalité
            month = date.month
            if month in [9, 10]:  # Pic de naissances 9 mois après les fêtes
                seasonal = 1.1
            else:
                seasonal = 1.0
            
            noise = np.random.normal(1, 0.07)
            births.append(base_births * seasonal * noise)
        
        return births
    
    def _simulate_nord_budget(self, dates):
        """Simule le budget du CHU Nord (en milliers d'euros)"""
        base_budget = 25000  # k€/mois
        
        budget = []
        for i, date in enumerate(dates):
            # Croissance annuelle d'environ 3%
            growth = 1 + 0.03 * (i / len(dates))
            noise = np.random.normal(1, 0.04)
            budget.append(base_budget * growth * noise)
        
        return budget
    
    def _simulate_nord_staff(self, dates):
        """Simule le nombre de personnel au CHU Nord"""
        base_staff = 3500  # personnels
        
        staff = []
        for i, date in enumerate(dates):
            # Croissance régulière
            growth = 1 + 0.005 * (i / len(dates))
            staff.append(base_staff * growth)
        
        return staff
    
    def _simulate_nord_beds(self, dates):
        """Simule le nombre de lits au CHU Nord"""
        base_beds = 1200  # lits
        
        beds = []
        for i, date in enumerate(dates):
            # Légère augmentation sur la période
            growth = 1 + 0.002 * (i / len(dates))
            beds.append(base_beds * growth)
        
        return beds
    
    def _simulate_sud_patients(self, dates):
        """Simule le nombre de patients pour le CHU Sud"""
        base_patients = 12000  # patients/mois
        
        patients = []
        for i, date in enumerate(dates):
            # Croissance plus rapide que le Nord (développement récent)
            growth = 1 + 0.008 * (i / len(dates))
            
            # Saisonnalité similaire
            month = date.month
            if month in [6, 7, 8]:  # Hiver austral
                seasonal = 1.1
            else:
                seasonal = 1.0
            
            noise = np.random.normal(1, 0.06)
            patients.append(base_patients * growth * seasonal * noise)
        
        return patients
    
    def _simulate_sud_emergencies(self, dates):
        """Simule les passages aux urgences du CHU Sud"""
        base_emergencies = 6000  # urgences/mois
        
        emergencies = []
        for date in dates:
            # Saisonnalité (pic en été avec la dengue)
            month = date.month
            if month in [1, 2, 3]:  # Été austral, saison des pluies
                seasonal = 1.3
            else:
                seasonal = 1.0
            
            noise = np.random.normal(1, 0.09)
            emergencies.append(base_emergencies * seasonal * noise)
        
        return emergencies
    
    def _simulate_sud_surgeries(self, dates):
        """Simule le nombre de chirurgies au CHU Sud"""
        base_surgeries = 900  # chirurgies/mois
        
        surgeries = []
        for i, date in enumerate(dates):
            # Croissance plus rapide que le Nord
            growth = 1 + 0.015 * (i / len(dates))
            noise = np.random.normal(1, 0.07)
            surgeries.append(base_surgeries * growth * noise)
        
        return surgeries
    
    def _simulate_sud_births(self, dates):
        """Simule le nombre de naissances au CHU Sud"""
        base_births = 400  # naissances/mois
        
        births = []
        for date in dates:
            # Légère saisonnalité
            month = date.month
            if month in [9, 10]:  # Pic de naissances 9 mois après les fêtes
                seasonal = 1.15
            else:
                seasonal = 1.0
            
            noise = np.random.normal(1, 0.08)
            births.append(base_births * seasonal * noise)
        
        return births
    
    def _simulate_sud_budget(self, dates):
        """Simule le budget du CHU Sud (en milliers d'euros)"""
        base_budget = 18000  # k€/mois
        
        budget = []
        for i, date in enumerate(dates):
            # Croissance plus rapide que le Nord
            growth = 1 + 0.035 * (i / len(dates))
            noise = np.random.normal(1, 0.05)
            budget.append(base_budget * growth * noise)
        
        return budget
    
    def _simulate_sud_staff(self, dates):
        """Simule le nombre de personnel au CHU Sud"""
        base_staff = 2800  # personnels
        
        staff = []
        for i, date in enumerate(dates):
            # Croissance plus rapide que le Nord
            growth = 1 + 0.008 * (i / len(dates))
            staff.append(base_staff * growth)
        
        return staff
    
    def _simulate_sud_beds(self, dates):
        """Simule le nombre de lits au CHU Sud"""
        base_beds = 900  # lits
        
        beds = []
        for i, date in enumerate(dates):
            # Croissance plus rapide que le Nord
            growth = 1 + 0.004 * (i / len(dates))
            beds.append(base_beds * growth)
        
        return beds
    
    def _simulate_dengue_cases(self, dates):
        """Simule les cas de dengue à La Réunion"""
        base_cases = 500  # cas/mois en base
        
        cases = []
        for date in dates:
            # Forte saisonnalité (été austral)
            month = date.month
            if month in [1, 2, 3, 4]:  # Saison des pluies
                seasonal = np.random.uniform(3, 8)  # Épidémies
            else:
                seasonal = np.random.uniform(0.2, 1.5)
            
            # Cycles épidémiques tous les 3-4 ans
            year = date.year
            if year in [2013, 2017, 2020, 2024]:
                epidemic = 1.5
            else:
                epidemic = 1.0
            
            cases.append(base_cases * seasonal * epidemic)
        
        return cases
    
    def _simulate_diabetes_cases(self, dates):
        """Simule les cas de diabète à La Réunion (problème majeur)"""
        base_cases = 2000  # cas/mois en base
        
        cases = []
        for i, date in enumerate(dates):
            # Croissance régulière liée au vieillissement et à l'alimentation
            growth = 1 + 0.01 * (i / len(dates))
            noise = np.random.normal(1, 0.03)
            cases.append(base_cases * growth * noise)
        
        return cases
    
    def _simulate_cardio_cases(self, dates):
        """Simule les cas de maladies cardiovasculaires à La Réunion"""
        base_cases = 1500  # cas/mois en base
        
        cases = []
        for i, date in enumerate(dates):
            # Croissance régulière
            growth = 1 + 0.007 * (i / len(dates))
            noise = np.random.normal(1, 0.04)
            cases.append(base_cases * growth * noise)
        
        return cases
    
    def _add_health_trends(self, df):
        """Ajoute des tendances sanitaires réalistes"""
        for i, row in df.iterrows():
            date = row['Date']
            year = date.year
            
            # Impact COVID-19 (2020-2021)
            if 2020 <= year <= 2021:
                if year == 2020 and date.month in [3, 4, 5]:
                    # Baisse des activités non urgentes, augmentation des urgences
                    df.loc[i, 'CHU_Nord_Chirurgies'] *= 0.6
                    df.loc[i, 'CHU_Sud_Chirurgies'] *= 0.6
                    df.loc[i, 'CHU_Nord_Urgences'] *= 1.1
                    df.loc[i, 'CHU_Sud_Urgences'] *= 1.1
            
            # Développement du CHU Sud (2015-2018)
            elif 2015 <= year <= 2018:
                df.loc[i, 'CHU_Sud_Budget'] *= 1.05
                df.loc[i, 'CHU_Sud_Personnel'] *= 1.03
            
            # Vieillissement de la population (augmentation constante)
            if year >= 2015:
                aging = 1 + 0.002 * (year - 2015)
                df.loc[i, 'CHU_Nord_Patients'] *= aging
                df.loc[i, 'CHU_Sud_Patients'] *= aging
            
            # Augmentation des maladies chroniques
            if year >= 2010:
                chronic_growth = 1 + 0.005 * (year - 2010)
                df.loc[i, 'Diabete_Cas'] *= chronic_growth
                df.loc[i, 'Cardio_Cas'] *= chronic_growth
    
    def create_health_analysis(self, df):
        """Crée une analyse complète des données hospitalières"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 20))
        
        # 1. Comparaison des patients entre CHU Nord et Sud
        ax1 = plt.subplot(3, 2, 1)
        self._plot_patient_comparison(df, ax1)
        
        # 2. Activité aux urgences
        ax2 = plt.subplot(3, 2, 2)
        self._plot_emergency_activity(df, ax2)
        
        # 3. Budget et ressources
        ax3 = plt.subplot(3, 2, 3)
        self._plot_budget_resources(df, ax3)
        
        # 4. Épidémiologie réunionnaise
        ax4 = plt.subplot(3, 2, 4)
        self._plot_epidemiology(df, ax4)
        
        # 5. Évolution annuelle comparative
        ax5 = plt.subplot(3, 2, 5)
        self._plot_yearly_comparison(df, ax5)
        
        # 6. Performances et indicateurs
        ax6 = plt.subplot(3, 2, 6)
        self._plot_performance_indicators(df, ax6)
        
        plt.suptitle(f'Analyse Comparative des CHU de La Réunion (2010-2024)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('chu_reunion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Générer les insights
        self._generate_health_insights(df)
    
    def _plot_patient_comparison(self, df, ax):
        """Plot de comparaison des patients entre CHU Nord et Sud"""
        ax.plot(df['Date'], df['CHU_Nord_Patients'], label='CHU Nord', 
               linewidth=2, color='#264653', alpha=0.8)
        ax.plot(df['Date'], df['CHU_Sud_Patients'], label='CHU Sud', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        
        ax.set_title('Nombre de Patients Traités par Mois', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Nombre de Patients')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_emergency_activity(self, df, ax):
        """Plot de l'activité aux urgences"""
        ax.plot(df['Date'], df['CHU_Nord_Urgences'], label='CHU Nord', 
               linewidth=2, color='#E76F51', alpha=0.8)
        ax.plot(df['Date'], df['CHU_Sud_Urgences'], label='CHU Sud', 
               linewidth=2, color='#F9A602', alpha=0.8)
        
        ax.set_title('Passages aux Urgences par Mois', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Nombre de Passages')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_budget_resources(self, df, ax):
        """Plot des budgets et ressources"""
        # Budgets
        ax.plot(df['Date'], df['CHU_Nord_Budget'], label='Budget CHU Nord', 
               linewidth=2, color='#6A0572', alpha=0.8)
        ax.plot(df['Date'], df['CHU_Sud_Budget'], label='Budget CHU Sud', 
               linewidth=2, color='#AB83A1', alpha=0.8)
        
        ax.set_title('Budget Mensuel (k€)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Budget (k€)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Personnel en second axe
        ax2 = ax.twinx()
        ax2.plot(df['Date'], df['CHU_Nord_Personnel'], label='Personnel CHU Nord', 
                linestyle='--', linewidth=1, color='#45B7D1', alpha=0.8)
        ax2.plot(df['Date'], df['CHU_Sud_Personnel'], label='Personnel CHU Sud', 
                linestyle='--', linewidth=1, color='#4ECDC4', alpha=0.8)
        ax2.set_ylabel('Nombre de Personnels')
    
    def _plot_epidemiology(self, df, ax):
        """Plot des données épidémiologiques"""
        ax.plot(df['Date'], df['Dengue_Cas'], label='Cas de Dengue', 
               linewidth=2, color='#FF6B6B', alpha=0.8)
        
        ax2 = ax.twinx()
        ax2.plot(df['Date'], df['Diabete_Cas'], label='Cas de Diabète', 
                linewidth=2, color='#5CAB7D', alpha=0.8)
        ax2.plot(df['Date'], df['Cardio_Cas'], label='Cas Cardio', 
                linewidth=2, color='#F9A602', alpha=0.8)
        
        ax.set_title('Épidémiologie à La Réunion', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cas de Dengue', color='#FF6B6B')
        ax2.set_ylabel('Cas Chroniques', color='#5CAB7D')
        ax.tick_params(axis='y', labelcolor='#FF6B6B')
        ax2.tick_params(axis='y', labelcolor='#5CAB7D')
        ax.grid(True, alpha=0.3)
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_yearly_comparison(self, df, ax):
        """Plot de l'évolution annuelle comparative"""
        df_yearly = df.copy()
        df_yearly['Year'] = df_yearly['Date'].dt.year
        
        yearly_data = df_yearly.groupby('Year').agg({
            'CHU_Nord_Patients': 'mean',
            'CHU_Sud_Patients': 'mean',
            'CHU_Nord_Budget': 'mean',
            'CHU_Sud_Budget': 'mean'
        })
        
        x = yearly_data.index
        width = 0.35
        
        ax.bar(x - width/2, yearly_data['CHU_Nord_Patients'], width, 
               label='CHU Nord', color='#264653', alpha=0.7)
        ax.bar(x + width/2, yearly_data['CHU_Sud_Patients'], width, 
               label='CHU Sud', color='#2A9D8F', alpha=0.7)
        
        ax.set_title('Patients Moyens par An', fontsize=12, fontweight='bold')
        ax.set_ylabel('Nombre de Patients')
        ax.set_xlabel('Année')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_performance_indicators(self, df, ax):
        """Plot des indicateurs de performance"""
        # Calcul des indicateurs
        df['Nord_Patients_Par_Personnel'] = df['CHU_Nord_Patients'] / df['CHU_Nord_Personnel']
        df['Sud_Patients_Par_Personnel'] = df['CHU_Sud_Patients'] / df['CHU_Sud_Personnel']
        
        df['Nord_Budget_Par_Patient'] = df['CHU_Nord_Budget'] / df['CHU_Nord_Patients']
        df['Sud_Budget_Par_Patient'] = df['CHU_Sud_Budget'] / df['CHU_Sud_Patients']
        
        ax.plot(df['Date'], df['Nord_Patients_Par_Personnel'], 
               label='CHU Nord (Patients/Personnel)', linewidth=2, color='#264653')
        ax.plot(df['Date'], df['Sud_Patients_Par_Personnel'], 
               label='CHU Sud (Patients/Personnel)', linewidth=2, color='#2A9D8F')
        
        ax.set_title('Indicateurs de Performance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Patients par Personnel')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Budget par patient en second axe
        ax2 = ax.twinx()
        ax2.plot(df['Date'], df['Nord_Budget_Par_Patient'], 
                label='CHU Nord (Budget/Patient)', linestyle='--', linewidth=1, color='#E76F51')
        ax2.plot(df['Date'], df['Sud_Budget_Par_Patient'], 
                label='CHU Sud (Budget/Patient)', linestyle='--', linewidth=1, color='#F9A602')
        ax2.set_ylabel('Budget par Patient (€)')
    
    def _generate_health_insights(self, df):
        """Génère des insights analytiques"""
        print(f"🏥 INSIGHTS ANALYTIQUES - CHU de La Réunion")
        print("=" * 60)
        
        # 1. Statistiques de base
        print("\n1. 📈 STATISTIQUES GÉNÉRALES (2010-2024):")
        total_nord_patients = df['CHU_Nord_Patients'].sum()
        total_sud_patients = df['CHU_Sud_Patients'].sum()
        avg_nord_budget = df['CHU_Nord_Budget'].mean()
        avg_sud_budget = df['CHU_Sud_Budget'].mean()
        
        print(f"Patients totaux CHU Nord: {total_nord_patients:,.0f}")
        print(f"Patients totaux CHU Sud: {total_sud_patients:,.0f}")
        print(f"Budget moyen mensuel CHU Nord: {avg_nord_budget:,.0f} k€")
        print(f"Budget moyen mensuel CHU Sud: {avg_sud_budget:,.0f} k€")
        
        # 2. Croissance
        print("\n2. 📊 TAUX DE CROISSANCE:")
        growth_nord = ((df['CHU_Nord_Patients'].iloc[-12:].mean() / 
                       df['CHU_Nord_Patients'].iloc[:12].mean()) - 1) * 100
        growth_sud = ((df['CHU_Sud_Patients'].iloc[-12:].mean() / 
                      df['CHU_Sud_Patients'].iloc[:12].mean()) - 1) * 100
        
        print(f"Croissance des patients CHU Nord: {growth_nord:.1f}%")
        print(f"Croissance des patients CHU Sud: {growth_sud:.1f}%")
        
        # 3. Comparaison des ressources
        print("\n3. 📋 COMPARAISON DES RESSOURCES:")
        nord_patients_per_staff = df['CHU_Nord_Patients'].mean() / df['CHU_Nord_Personnel'].mean()
        sud_patients_per_staff = df['CHU_Sud_Patients'].mean() / df['CHU_Sud_Personnel'].mean()
        
        nord_budget_per_patient = df['CHU_Nord_Budget'].mean() / df['CHU_Nord_Patients'].mean()
        sud_budget_per_patient = df['CHU_Sud_Budget'].mean() / df['CHU_Sud_Patients'].mean()
        
        print(f"Patients par personnel CHU Nord: {nord_patients_per_staff:.1f}")
        print(f"Patients par personnel CHU Sud: {sud_patients_per_staff:.1f}")
        print(f"Budget par patient CHU Nord: {nord_budget_per_patient:.1f} k€")
        print(f"Budget par patient CHU Sud: {sud_budget_per_patient:.1f} k€")
        
        # 4. Défis sanitaires
        print("\n4. 🦟 DÉFIS SANITAires:")
        avg_dengue = df['Dengue_Cas'].mean()
        avg_diabetes = df['Diabete_Cas'].mean()
        avg_cardio = df['Cardio_Cas'].mean()
        
        print(f"Cas moyens de dengue/mois: {avg_dengue:.0f}")
        print(f"Cas moyens de diabète/mois: {avg_diabetes:.0f}")
        print(f"Cas moyens de cardio/mois: {avg_cardio:.0f}")
        
        # 5. Recommandations
        print("\n5. 💡 RECOMMANDATIONS STRATÉGIQUES:")
        print("• Renforcer la prévention contre la dengue et les maladies chroniques")
        print("• Optimiser la répartition des ressources entre CHU Nord et Sud")
        print("• Développer la télémédecine pour les zones isolées")
        print("• Investir dans la formation du personnel médical local")
        print("• Adapter les infrastructures au vieillissement de la population")
        print("• Renforcer la coordination entre les deux CHU pour une meilleure complémentarité")

def main():
    """Fonction principale"""
    print("🏥 ANALYSE COMPARATIVE DES CHU DE LA RÉUNION")
    print("=" * 60)
    
    # Initialiser l'analyseur
    analyzer = CHUReunionAnalyzer()
    
    # Générer les données
    health_data = analyzer.generate_health_data()
    
    # Sauvegarder les données
    output_file = 'chu_reunion_health_data.csv'
    health_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(health_data[['Date', 'CHU_Nord_Patients', 'CHU_Sud_Patients', 'CHU_Nord_Budget', 'CHU_Sud_Budget']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse hospitalière...")
    analyzer.create_health_analysis(health_data)
    
    print(f"\n✅ Analyse des CHU de La Réunion terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("📦 Données: Patients, urgences, budgets, personnel, épidémiologie")

if __name__ == "__main__":
    main()