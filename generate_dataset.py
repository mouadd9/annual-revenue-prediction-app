# Configuration pour affichage des graphiques dans le notebook
# %matplotlib inline

# Bibliothèques pour la manipulation et l'analyse des données
import numpy as np
import pandas as pd
from datetime import datetime
import random

# Pour assurer la reproductibilité
np.random.seed(42)
random.seed(42)

# Définition des constantes basées sur les exigences du projet
# Statistiques de revenus du HCP (Haut-Commissariat au Plan)
REVENU_MOYEN_GLOBAL = 21949  # DH/an
REVENU_MOYEN_URBAIN = 26988  # DH/an
REVENU_MOYEN_RURAL = 12862   # DH/an

# Statistiques de distribution des revenus
PCT_BELOW_MEAN_GLOBAL = 0.718  # 71.8% sous la moyenne
PCT_BELOW_MEAN_URBAIN = 0.659  # 65.9% sous la moyenne en milieu urbain
PCT_BELOW_MEAN_RURAL = 0.854   # 85.4% sous la moyenne en milieu rural

# Taille de l'échantillon
N_SAMPLES = 40000


def generate_demographics(n_samples=N_SAMPLES):
    """
    Génère des données démographiques réalistes pour n_samples individus marocains.
    
    Args:
        n_samples (int): Nombre d'individus à générer
        
    Returns:
        pandas.DataFrame: DataFrame contenant les caractéristiques démographiques
    """
    demographics = pd.DataFrame()
    
    # 1. Âge : le revenu a tendance d'augmenter lorsqu'on se rapproche de l'âge de retraite
    demographics['age'] = np.random.normal(loc=35, scale=12, size=n_samples).astype(int) 
    demographics['age'] = np.clip(demographics['age'], 18, 80)

    # 2. Catégorie d'âge : jeune, adulte, sénior et âgé
    demographics['categorie_age'] = pd.cut(
        demographics['age'], 
        bins=[0, 25, 45, 65, 100],
        labels=['jeune', 'adulte', 'sénior', 'âgé']
    )

    # 3. Urbain/Rural : les personnes issues du milieu urbain ont tendance à gagner plus
    # que les personnes du milieu rural (63% urbain, selon les statistiques marocaines)
    demographics['milieu'] = np.random.choice(['urbain', 'rural'], size=n_samples, p=[0.63, 0.37])

    # 4. Sexe : le revenu moyen des hommes est plus élevé que celui des femmes
    demographics['sexe'] = np.random.choice(['homme', 'femme'], size=n_samples, p=[0.51, 0.49])

    # 5. Niveau d'éducation : quatre niveaux (corrélé avec l'âge)
    education_levels = ['sans_niveau', 'fondamental', 'secondaire', 'supérieur']
    
    # Probabilités d'éducation selon la catégorie d'âge
    education_probs = {
      'jeune': [0.05, 0.25, 0.40, 0.30],  # Les jeunes ont plus souvent une éducation secondaire ou supérieure
      'adulte': [0.15, 0.35, 0.30, 0.20],
      'sénior': [0.30, 0.40, 0.20, 0.10],
      'âgé': [0.50, 0.30, 0.15, 0.05]     # Les personnes âgées ont plus souvent un niveau d'éducation faible
    }
    
    demographics['niveau_education'] = demographics.apply(
      lambda row: np.random.choice(education_levels, p=education_probs[row['categorie_age']]),
      axis=1
    )
    
    # 6. Années d'expérience : calculées en fonction de l'âge et du niveau d'éducation
    demographics['annees_experience'] = demographics.apply(
        lambda row: max(0, min(row['age'] - 18 - (4 if row['niveau_education'] == 'supérieur' else 
                                            2 if row['niveau_education'] == 'secondaire' else 
                                            0 if row['niveau_education'] == 'fondamental' else 0),
                      row['age'] - 15)) + np.random.randint(-2, 3),
        axis=1
    )
    demographics['annees_experience'] = np.clip(demographics['annees_experience'], 0, 50)
    
    # 7. État matrimonial : corrélé avec l'âge
    marital_status = ['célibataire', 'marié', 'divorcé', 'veuf']
    marital_probs = {
        'jeune': [0.80, 0.18, 0.02, 0.00],
        'adulte': [0.25, 0.65, 0.08, 0.02],
        'sénior': [0.10, 0.70, 0.10, 0.10],
        'âgé': [0.05, 0.55, 0.10, 0.30]
    }
    
    demographics['etat_matrimonial'] = demographics.apply(
        lambda row: np.random.choice(marital_status, p=marital_probs[row['categorie_age']]), 
        axis=1
    )

    # 8. Catégorie socioprofessionnelle : corrélée avec le niveau d'éducation
    socio_prof_categories = ['Groupe_1', 'Groupe_2', 'Groupe_3', 'Groupe_4', 'Groupe_5', 'Groupe_6']
    socio_prof_probs = {
        'sans_niveau': [0.01, 0.03, 0.10, 0.25, 0.25, 0.36],
        'fondamental': [0.03, 0.10, 0.15, 0.22, 0.30, 0.20],
        'secondaire': [0.10, 0.25, 0.20, 0.10, 0.25, 0.10],
        'supérieur': [0.40, 0.35, 0.15, 0.02, 0.05, 0.03]
    }
    
    demographics['categorie_socioprofessionnelle'] = demographics.apply(
        lambda row: np.random.choice(socio_prof_categories, p=socio_prof_probs[row['niveau_education']]), 
        axis=1
    )

    # 9. Possession de biens : corrélée avec la catégorie socioprofessionnelle
    group_weights = {
        'Groupe_1': 0.9, 'Groupe_2': 0.7, 'Groupe_3': 0.5, 
        'Groupe_4': 0.3, 'Groupe_5': 0.2, 'Groupe_6': 0.1
    }
    
    demographics['possession_voiture'] = demographics.apply(
        lambda row: np.random.choice([1, 0], p=[group_weights[row['categorie_socioprofessionnelle']], 
                                               1-group_weights[row['categorie_socioprofessionnelle']]]),
        axis=1
    )
    
    demographics['possession_logement'] = demographics.apply(
        lambda row: np.random.choice([1, 0], p=[np.clip(group_weights[row['categorie_socioprofessionnelle']]*0.8, 0, 1),
                                              1-np.clip(group_weights[row['categorie_socioprofessionnelle']]*0.8, 0, 1)]),
        axis=1
    )
    
    demographics['possession_terrain'] = demographics.apply(
        lambda row: np.random.choice([1, 0], p=[np.clip(group_weights[row['categorie_socioprofessionnelle']]*0.4, 0, 1),
                                              1-np.clip(group_weights[row['categorie_socioprofessionnelle']]*0.4, 0, 1)]),
        axis=1
    )

    # 10. Caractéristiques supplémentaires (au moins 3 selon les exigences du projet)
    
    # a. Nombre d'enfants
    demographics['nombre_enfants'] = demographics.apply(
        lambda row: np.random.poisson(2) if row['etat_matrimonial'] in ['marié', 'divorcé', 'veuf'] else
                   (np.random.binomial(1, 0.05) if row['etat_matrimonial'] == 'célibataire' else 0),
        axis=1
    )
    
    # b. Taille du ménage
    demographics['taille_menage'] = demographics.apply(
        lambda row: row['nombre_enfants'] + (2 if row['etat_matrimonial'] == 'marié' else 1),
        axis=1
    )
    
    # c. Accès à Internet (plus courant en milieu urbain et avec un niveau d'éducation élevé)
    urban_factor = demographics['milieu'].apply(lambda x: 0.9 if x == 'urbain' else 0.4)
    education_factor = demographics['niveau_education'].apply(
        lambda x: 0.95 if x == 'supérieur' else 0.8 if x == 'secondaire' else 0.5 if x == 'fondamental' else 0.2
    )
    
    internet_prob = np.clip((urban_factor + education_factor) / 2, 0, 1)
    
    demographics['acces_internet'] = demographics.apply(
        lambda row: np.random.choice([1, 0], p=[internet_prob[row.name], 1-internet_prob[row.name]]),
        axis=1
    )
    
    return demographics

def generate_income(demographics_df):
    """
    Génère des revenus annuels basés sur les caractéristiques démographiques.
    Les revenus suivent une distribution réaliste tout en respectant les contraintes
    statistiques du HCP.
    
    Args:
        demographics_df (pandas.DataFrame): DataFrame contenant les caractéristiques démographiques
        
    Returns:
        numpy.ndarray: Tableau des revenus annuels générés
    """
    import numpy as np
    from scipy import stats
    
    # Création d'une copie pour éviter les modifications du DataFrame original
    df = demographics_df.copy()
    
    # Séparation urbain/rural pour un traitement indépendant
    urban_mask = df['milieu'] == 'urbain'
    rural_mask = ~urban_mask
    
    # Initialisation du vecteur de revenus
    income = np.zeros(len(df))
    
    # Fonction pour générer les revenus pour un segment spécifique
    def generate_segment_income(segment_mask, mean_target, pct_below_mean):
        segment_size = segment_mask.sum()
        segment_df = df.loc[segment_mask].copy()
        
        # Calcul des paramètres optimaux de la distribution log-normale
        # Dans une log-normale, la proportion sous la moyenne est liée au sigma
        # Nous devons trouver le sigma qui donne exactement le pourcentage cible
        
        # Pour une log-normale, le pourcentage sous la moyenne est:
        # P(X < μ) = Φ(σ/sqrt(2)), où Φ est la CDF de la loi normale standard
        
        # Inversement, à partir du pourcentage, on peut calculer sigma:
        # sigma = sqrt(2) * Φ^(-1)(pct_below_mean)
        # où Φ^(-1) est l'inverse de la CDF de la loi normale standard
        
        # Calcul de sigma optimal
        sigma = np.sqrt(2) * stats.norm.ppf(pct_below_mean)
        
        # Calcul de mu (tel que la moyenne de la distribution log-normale soit mean_target)
        # Pour une log-normale, E[X] = exp(mu + sigma²/2)
        # D'où: mu = ln(mean_target) - sigma²/2
        mu = np.log(mean_target) - (sigma**2)/2
        
        # Génération du revenu de base avec distribution log-normale optimisée
        base_income = np.random.lognormal(mean=mu, sigma=sigma, size=segment_size)
        
        # Vérification (uniquement pour le développement, peut être commenté en production)
        actual_mean = base_income.mean()
        actual_pct_below = (base_income < mean_target).mean()
        print(f"Distribution - Mean: {actual_mean:.2f} (target: {mean_target:.2f}), "
              f"Pct below: {actual_pct_below*100:.1f}% (target: {pct_below_mean*100:.1f}%)")
        
        # Matrice de modifications basée sur les caractéristiques démographiques
        modifiers = np.ones(segment_size)
        
        # --- Application des facteurs influençant le revenu ---
        
        # 1. Niveau d'éducation
        education_impact = {
            'sans_niveau': 0.75,
            'fondamental': 0.9,
            'secondaire': 1.1,
            'supérieur': 1.4
        }
        for edu, impact in education_impact.items():
            modifiers[segment_df['niveau_education'] == edu] *= impact
        
        # 2. Années d'expérience (effet modéré avec plafond)
        exp_values = segment_df['annees_experience'].values
        exp_modifier = 1 + 0.01 * exp_values
        exp_modifier = np.minimum(exp_modifier, 1.3)
        modifiers *= exp_modifier
        
        # 3. Catégorie socioprofessionnelle
        socio_prof_impact = {
            'Groupe_1': 1.5,  # Cadres supérieurs
            'Groupe_2': 1.3,  # Cadres moyens
            'Groupe_3': 1.1,  # Inactifs avec revenus
            'Groupe_4': 0.9,  # Agriculteurs
            'Groupe_5': 0.85, # Artisans, ouvriers
            'Groupe_6': 0.8   # Travailleurs non qualifiés
        }
        for group, impact in socio_prof_impact.items():
            modifiers[segment_df['categorie_socioprofessionnelle'] == group] *= impact
        
        # 4. Sexe (facteurs modérés)
        modifiers[segment_df['sexe'] == 'homme'] *= 1.1
        modifiers[segment_df['sexe'] == 'femme'] *= 0.9
        
        # 5. Âge (fonction en cloche)
        age_values = segment_df['age'].values
        age_modifier = 0.85 + 0.3 * np.exp(-0.002 * (age_values - 45)**2)
        modifiers *= age_modifier
        
        # Appliquer les multiplicateurs en préservant les rangs
        # Pour ce faire, nous réorganisons les revenus de base selon les rangs des modificateurs
        
        # Trier les revenus de base
        sorted_base_income = np.sort(base_income)
        
        # Obtenir les rangs des modificateurs
        modifier_ranks = stats.rankdata(modifiers, method='ordinal') - 1
        
        # Réorganiser les revenus de base selon les rangs des modificateurs
        # Cela préserve la distribution tout en créant une corrélation avec les facteurs démographiques
        income_with_factors = np.zeros_like(base_income)
        for i, rank in enumerate(modifier_ranks):
            income_with_factors[i] = sorted_base_income[rank]
        
        # Réajuster la moyenne pour compenser les effets de la réorganisation
        adjusted_income = income_with_factors * (mean_target / income_with_factors.mean())
        
        return adjusted_income
    
    # Génération des revenus pour la population urbaine
    if urban_mask.sum() > 0:
        urban_income = generate_segment_income(urban_mask, REVENU_MOYEN_URBAIN, PCT_BELOW_MEAN_URBAIN)
        income[urban_mask] = urban_income
    
    # Génération des revenus pour la population rurale
    if rural_mask.sum() > 0:
        rural_income = generate_segment_income(rural_mask, REVENU_MOYEN_RURAL, PCT_BELOW_MEAN_RURAL)
        income[rural_mask] = rural_income
    
    return income


def ajouter_valeurs_manquantes(df, taux_manquant=0.02):
    """
    Ajoute des valeurs manquantes de manière structurée dans le dataset.
    
    Cette fonction utilise trois patterns différents de valeurs manquantes:
    1. MCAR (Missing Completely At Random): valeurs manquantes aléatoires
    2. MAR (Missing At Random): manquantes conditionnelles à d'autres variables
    3. MNAR (Missing Not At Random): manquantes liées à la variable elle-même
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        taux_manquant (float): Taux global de valeurs manquantes à introduire
        
    Returns:
        pandas.DataFrame: DataFrame avec valeurs manquantes ajoutées
    """
    df_missing = df.copy()
    
    # Pattern 1: Données manquantes chez les personnes âgées (MAR)
    # Justification: Les données d'éducation des personnes âgées sont souvent moins bien documentées
    elderly_mask = df_missing['categorie_age'] == 'âgé'
    if sum(elderly_mask) > 0:
        missing_in_elderly = np.random.random(size=sum(elderly_mask)) < 0.15  # 15% des personnes âgées
        elderly_indices = df_missing[elderly_mask].index[missing_in_elderly]
        df_missing.loc[elderly_indices, 'niveau_education'] = np.nan
        print(f"Valeurs manquantes ajoutées dans 'niveau_education' pour {len(elderly_indices)} personnes âgées")
    
    # Pattern 2: Données manquantes pour les hauts revenus (MNAR)
    # Justification: Les personnes à haut revenu divulguent moins souvent leurs biens
    if 'revenu_annuel' in df_missing.columns:
        high_income_mask = df_missing['revenu_annuel'] > df_missing['revenu_annuel'].quantile(0.8)
        high_income_indices = df_missing[high_income_mask].index
        missing_property_mask = np.random.random(size=len(high_income_indices)) < 0.2  # 20% des hauts revenus
        missing_property_indices = high_income_indices[missing_property_mask]
        df_missing.loc[missing_property_indices, 'possession_terrain'] = np.nan
        print(f"Valeurs manquantes ajoutées dans 'possession_terrain' pour {len(missing_property_indices)} personnes à haut revenu")
    
    # Pattern 3: Données manquantes complètement aléatoires (MCAR)
    # Justification: Simule des erreurs de collecte aléatoires indépendantes des variables
    random_missing = np.random.random(size=len(df_missing)) < 0.03  # 3% aléatoire
    random_indices = df_missing.index[random_missing]
    df_missing.loc[random_indices, 'annees_experience'] = np.nan
    print(f"Valeurs manquantes ajoutées dans 'annees_experience' pour {len(random_indices)} individus aléatoires")
    
    return df_missing

def ajouter_valeurs_aberrantes(df, taux_aberrant=0.01):
    """
    Ajoute des valeurs aberrantes structurées au dataset.
    
    Cette fonction introduit trois types d'aberrations:
    1. Âges impossibles: simule des erreurs de saisie
    2. Expérience incohérente: expérience supérieure à l'âge (impossible logiquement)
    3. Revenus négatifs: simule des erreurs de signe
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        taux_aberrant (float): Taux global de valeurs aberrantes à introduire
        
    Returns:
        pandas.DataFrame: DataFrame avec valeurs aberrantes ajoutées
    """
    df_outliers = df.copy()
    
    # Aberration 1: Âges impossibles (erreur de saisie - ajout d'un 1 devant)
    # Justification: Simule une erreur de saisie courante (appui accidentel sur une touche)
    age_error_mask = np.random.random(size=len(df_outliers)) < taux_aberrant
    age_error_indices = df_outliers.index[age_error_mask]
    df_outliers.loc[age_error_indices, 'age'] = 100 + df_outliers.loc[age_error_indices, 'age']
    print(f"Âges aberrants ajoutés pour {len(age_error_indices)} individus")
    
    # Aberration 2: Expérience professionnelle incohérente (erreur logique)
    # Justification: Incohérence logique permettant de démontrer l'importance des vérifications croisées
    exp_error_mask = np.random.random(size=len(df_outliers)) < taux_aberrant
    exp_error_indices = df_outliers.index[exp_error_mask]
    df_outliers.loc[exp_error_indices, 'annees_experience'] = df_outliers.loc[exp_error_indices, 'age'] + \
                                                          np.random.randint(1, 10, size=len(exp_error_indices))
    print(f"Expériences aberrantes ajoutées pour {len(exp_error_indices)} individus")
    
    # Aberration 3: Revenus négatifs (erreur de signe)
    # Justification: Erreur de signe, permettant d'illustrer la détection par contrainte de domaine
    if 'revenu_annuel' in df_outliers.columns:
        income_error_mask = np.random.random(size=len(df_outliers)) < taux_aberrant/2  # moins fréquent
        income_error_indices = df_outliers.index[income_error_mask]
        df_outliers.loc[income_error_indices, 'revenu_annuel'] = -1 * df_outliers.loc[income_error_indices, 'revenu_annuel']
        print(f"Revenus négatifs ajoutés pour {len(income_error_indices)} individus")
    
    return df_outliers

def ajouter_colonnes_redondantes(df):
    """
    Ajoute des colonnes redondantes au dataset.
    
    Cette fonction crée trois types de redondances:
    1. Redondance parfaite: copie exacte d'une colonne
    2. Redondance avec transformation: même information sous un format différent
    3. Redondance dérivée: information calculée à partir d'autres colonnes
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        
    Returns:
        pandas.DataFrame: DataFrame avec colonnes redondantes ajoutées
    """
    df_redundant = df.copy()
    
    # Redondance 1: Copie exacte (redondance parfaite)
    # Justification: Démontre la détection de colonnes identiques
    df_redundant['age_ans'] = df_redundant['age']
    print("Colonne redondante 'age_ans' ajoutée (copie exacte de 'age')")
    
    # Redondance 2: Même information, codage différent (redondance de format)
    # Justification: Illustre la redondance moins évidente, nécessitant analyse
    df_redundant['homme_femme'] = df_redundant['sexe']
    print("Colonne redondante 'homme_femme' ajoutée (même information que 'sexe')")
    
    # Redondance 3: Information dérivée (redondance calculée)
    # Justification: Montre une redondance fonctionnelle, détectable par corrélation
    if 'revenu_annuel' in df_redundant.columns:
        df_redundant['revenu_mensuel'] = df_redundant['revenu_annuel'] / 12
        print("Colonne redondante 'revenu_mensuel' ajoutée (dérivée de 'revenu_annuel')")
    
    # Redondance 4: Recodage d'une variable catégorielle (redondance de représentation)
    # Justification: Illustre la redondance par transformation d'encodage
    df_redundant['est_urbain'] = df_redundant['milieu'].map({'urbain': 1, 'rural': 0})
    print("Colonne redondante 'est_urbain' ajoutée (recodage de 'milieu')")
    
    return df_redundant

def ajouter_colonnes_non_pertinentes(df):
    """
    Ajoute des colonnes non pertinentes au dataset.
    
    Cette fonction ajoute trois types de colonnes non pertinentes:
    1. Identifiants: colonnes purement techniques sans valeur prédictive
    2. Métadonnées: informations sur le processus de collecte sans lien avec le revenu
    3. Variables trompeuses: semblent pertinentes mais n'ont pas de lien causal avec le revenu
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        
    Returns:
        pandas.DataFrame: DataFrame avec colonnes non pertinentes ajoutées
    """
    df_irrelevant = df.copy()
    
    # Non pertinence 1: Identifiants uniques (sans valeur prédictive)
    # Justification: Les identifiants n'ont aucune valeur prédictive et doivent être écartés
    df_irrelevant['id_unique'] = np.arange(len(df_irrelevant))
    print("Colonne non pertinente 'id_unique' ajoutée")
    
    # Non pertinence 2: Métadonnées de collecte (sans lien avec le phénomène étudié)
    # Justification: Informations sur le processus sans lien avec le revenu
    df_irrelevant['date_saisie'] = pd.Timestamp('today').strftime('%Y-%m-%d')
    df_irrelevant['operateur_saisie'] = np.random.choice(
        ['OP001', 'OP002', 'OP003', 'OP004', 'OP005'],
        size=len(df_irrelevant)
    )
    print("Colonnes non pertinentes 'date_saisie' et 'operateur_saisie' ajoutées")
    
    # Non pertinence 3: Variables trompeuses (corrélation sans causalité)
    # Justification: Variable qui semble liée mais sans relation causale avec le revenu
    # Ici, la couleur préférée est liée au sexe, qui influence le revenu, créant une corrélation trompeuse
    df_irrelevant['couleur_preferee'] = np.where(
        df_irrelevant['sexe'] == 'homme',
        np.random.choice(['bleu', 'vert', 'rouge', 'noir'], p=[0.4, 0.3, 0.2, 0.1], size=len(df_irrelevant)),
        np.random.choice(['rouge', 'vert', 'bleu', 'jaune'], p=[0.4, 0.3, 0.2, 0.1], size=len(df_irrelevant))
    )
    print("Colonne non pertinente 'couleur_preferee' ajoutée")
    
    # Non pertinence 4: Codes géographiques aléatoires (sans structure pertinente)
    # Justification: Codes aléatoires qui ne contiennent aucune information sur la répartition géographique réelle
    df_irrelevant['code_postal'] = np.random.randint(10000, 99999, size=len(df_irrelevant))
    print("Colonne non pertinente 'code_postal' ajoutée")
    
    return df_irrelevant
    
def verify_income_constraints(demographics_df):
    """
    Vérifie si les contraintes sur les revenus sont respectées et affiche un tableau comparatif.
    
    Args:
        demographics_df (pandas.DataFrame): DataFrame avec les données démographiques et de revenu
    """
    # Calcul des statistiques réelles
    actual_mean_global = demographics_df['revenu_annuel'].mean()
    
    urban_mask = demographics_df['milieu'] == 'urbain'
    rural_mask = ~urban_mask
    
    actual_mean_urbain = demographics_df.loc[urban_mask, 'revenu_annuel'].mean()
    actual_mean_rural = demographics_df.loc[rural_mask, 'revenu_annuel'].mean()
    
    actual_below_mean_global = (demographics_df['revenu_annuel'] < REVENU_MOYEN_GLOBAL).mean()
    actual_below_mean_urbain = (demographics_df.loc[urban_mask, 'revenu_annuel'] < REVENU_MOYEN_URBAIN).mean()
    actual_below_mean_rural = (demographics_df.loc[rural_mask, 'revenu_annuel'] < REVENU_MOYEN_RURAL).mean()
    
    # Création des données de comparaison
    data = {
        'Statistique': [
            'Revenu moyen global (DH/an)',
            'Revenu moyen urbain (DH/an)',
            'Revenu moyen rural (DH/an)',
            '% global < moyenne',
            '% urbain < moyenne',
            '% rural < moyenne'
        ],
        'Cible': [
            f"{REVENU_MOYEN_GLOBAL:.2f}",
            f"{REVENU_MOYEN_URBAIN:.2f}",
            f"{REVENU_MOYEN_RURAL:.2f}",
            f"{PCT_BELOW_MEAN_GLOBAL*100:.1f}%",
            f"{PCT_BELOW_MEAN_URBAIN*100:.1f}%",
            f"{PCT_BELOW_MEAN_RURAL*100:.1f}%"
        ],
        'Résultat': [
            f"{actual_mean_global:.2f}",
            f"{actual_mean_urbain:.2f}",
            f"{actual_mean_rural:.2f}",
            f"{actual_below_mean_global*100:.1f}%",
            f"{actual_below_mean_urbain*100:.1f}%",
            f"{actual_below_mean_rural*100:.1f}%"
        ],
        'Écart': [
            f"{actual_mean_global - REVENU_MOYEN_GLOBAL:.2f}",
            f"{actual_mean_urbain - REVENU_MOYEN_URBAIN:.2f}",
            f"{actual_mean_rural - REVENU_MOYEN_RURAL:.2f}",
            f"{(actual_below_mean_global - PCT_BELOW_MEAN_GLOBAL)*100:.1f}%",
            f"{(actual_below_mean_urbain - PCT_BELOW_MEAN_URBAIN)*100:.1f}%",
            f"{(actual_below_mean_rural - PCT_BELOW_MEAN_RURAL)*100:.1f}%"
        ]
    }
    
    # Création du DataFrame pour l'affichage
    comparison_df = pd.DataFrame(data)
    
    # Affichage du tableau avec mise en forme
    from IPython.display import display, HTML
    
    # Application de la mise en forme pour mettre en évidence les écarts
    def highlight_discrepancies(val):
        if isinstance(val, str):
            if '%' in val:
                if float(val.strip('%')) > 1.0 or float(val.strip('%')) < -1.0:
                    return 'background-color: yellow'
            else:
                if float(val) > 100 or float(val) < -100:
                    return 'background-color: yellow'
        return ''
    
    # Application de la mise en forme et affichage
    styled_df = comparison_df.style.map(highlight_discrepancies, subset=['Écart'])
    
    # Ajout d'un titre et affichage
    print("\n=== Vérification des contraintes statistiques ===\n")
    display(styled_df)
    
    # Ajout d'une évaluation sommaire
    mean_errors = [abs(actual_mean_global - REVENU_MOYEN_GLOBAL),
                  abs(actual_mean_urbain - REVENU_MOYEN_URBAIN),
                  abs(actual_mean_rural - REVENU_MOYEN_RURAL)]
    
    pct_errors = [abs(actual_below_mean_global - PCT_BELOW_MEAN_GLOBAL),
                 abs(actual_below_mean_urbain - PCT_BELOW_MEAN_URBAIN),
                 abs(actual_below_mean_rural - PCT_BELOW_MEAN_RURAL)]
    
    if max(mean_errors) < 100 and max(pct_errors) < 0.02:
        print("\n✓ Les contraintes statistiques sont bien respectées!")
    else:
        print("\n⚠ Certaines contraintes statistiques ne sont pas parfaitement respectées.")
        print("   Vous pourriez ajuster les paramètres de génération des revenus.")
    
    # Affichage de statistiques supplémentaires sur le jeu de données
    print(f"\nNombre total d'enregistrements: {len(demographics_df)}")
    print(f"Proportion urbain/rural: {sum(urban_mask)/len(demographics_df)*100:.1f}% / {sum(rural_mask)/len(demographics_df)*100:.1f}%")

def main():
    # Génération des données démographiques
    demographics_df = generate_demographics()
    # Génération des revenus initiaux
    initial_income = generate_income(demographics_df)
    # Ajustement des revenus pour respecter les contraintes statistiques
    # Ajout des revenus au DataFrame
    demographics_df['revenu_annuel'] = initial_income
    
    # Application des problèmes de qualité des données
    print("=== Ajout de problèmes de qualité au dataset ===")

    # Étape 1: Ajout de valeurs manquantes
    print("\n--- Ajout de valeurs manquantes ---")
    demographics_df_with_issues = ajouter_valeurs_manquantes(demographics_df)
    # Étape 2: Ajout de valeurs aberrantes
    print("\n--- Ajout de valeurs aberrantes ---")
    demographics_df_with_issues = ajouter_valeurs_aberrantes(demographics_df_with_issues)
    # Étape 3: Ajout de colonnes redondantes
    print("\n--- Ajout de colonnes redondantes ---")
    demographics_df_with_issues = ajouter_colonnes_redondantes(demographics_df_with_issues)
    # Étape 4: Ajout de colonnes non pertinentes
    print("\n--- Ajout de colonnes non pertinentes ---")
    demographics_df_with_issues = ajouter_colonnes_non_pertinentes(demographics_df_with_issues)

    # Affichage d'un aperçu des données avec problèmes de qualité
    print("\nAperçu du jeu de données avec problèmes de qualité ajoutés :")
    demographics_df_with_issues.sample(10)
    
    # Sauvegarde du jeu de données dans un fichier CSV
    demographics_df_with_issues.to_csv('dataset_revenu_marocains.csv', index=False)
    print(f"Jeu de données sauvegardé dans 'dataset_revenu_marocains.csv' avec {len(demographics_df_with_issues)} enregistrements et {len(demographics_df_with_issues.columns)} colonnes.")

    # Vérification des contraintes statistiques
    # verify_income_constraints(demographics_df)
    
if __name__ == "__main__":
    main()