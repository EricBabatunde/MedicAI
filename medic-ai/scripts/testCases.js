const testCases = [
    { domain: 'general_medicine', query: 'adult male with severe sudden chest pain radiating to left arm and sweating', target_phrase: 'acute coronary syndrome' },
    { domain: 'minor_surgery', query: 'drainage of a large fluctuant cutaneous abscess with local anesthesia', target_phrase: 'incision and drainage' },
    { domain: 'obstetrics', query: 'pregnant 24yo bright red vaginal bleeding third trimester', target_phrase: 'placenta praevia' },
    { domain: 'neonatology', query: 'newborn born at 34 weeks struggling to breathe with grunting and nasal flaring', target_phrase: 'respiratory distress' },
    { domain: 'infectious_disease', query: 'patient presenting with cyclical high fever, chills, and profuse sweating after travel', target_phrase: 'uncomplicated malaria' },
    { domain: 'pharmacology', query: 'loading dose of intravenous magnesium sulfate for severe eclampsia', target_phrase: '4 g of magnesium sulfate' },
    { domain: 'emergency_medicine', query: 'management of anaphylactic shock after insect sting with severe stridor', target_phrase: 'epinephrine' },
    { domain: 'orthopaedics', query: 'closed reduction and splinting of an uncomplicated distal radius fracture', target_phrase: 'plaster cast' },
    { domain: 'chronic_care', query: 'long term management and dietary advice for type 2 diabetes mellitus', target_phrase: 'glycaemic control' },
    { domain: 'mental_health', query: 'management of acute severe agitation and combative behavior in a psychotic patient', target_phrase: 'haloperidol' },
    { domain: 'dermatology', query: 'treatment for widespread extremely itchy rash worse at night with burrows', target_phrase: 'scabies' },
    { domain: 'nutrition', query: 'steps for calculating F-75 therapeutic milk for severe acute malnutrition', target_phrase: 'F-75' },
    { domain: 'anaesthesia', query: 'safe calculation of ketamine induction dose for a 50kg adult', target_phrase: 'ketamine induction' },
    { domain: 'ophthalmology', query: 'treatment for newborn with bilateral purulent eye discharge conjunctivitis', target_phrase: 'tetracycline eye' },
    { domain: 'dental', query: 'management of severe dental abscess with facial swelling', target_phrase: 'tooth extraction' },
    { domain: 'ent', query: 'child with foul smelling purulent discharge from one ear for 3 weeks', target_phrase: 'suppurative otitis media' }
];

module.exports = testCases;