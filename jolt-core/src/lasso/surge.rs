use crate::{
    field::JoltField,
    jolt::vm::{JoltCommitments, JoltPolynomials, ProverDebugInfo},
    poly::{
        compact_polynomial::{CompactPolynomial, SmallScalar},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    subprotocols::grand_product::BatchedDenseGrandProduct,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::marker::{PhantomData, Sync};
//NoExogenousOpenings is a zero-sized type, meaning it takes no memory. 
//StructuredPolynomialData is a trait that aims to organize polynomial data in a way that is easy to read and write
//VerifierComputedOpening represents the openings that the verifier can compute themselves. 
use super::memory_checking::{
    Initializable, NoExogenousOpenings, StructuredPolynomialData, VerifierComputedOpening,
};
use crate::{
    jolt::instruction::JoltInstruction,
    //MemoryCheckingProof defines what a proof contains, MemoryCheckingProver defines what a prover needs to do, 
    // MemoryCheckingVerifier defines what a verifier needs to do
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{
        commitment::commitment_scheme::CommitmentScheme, dense_mlpoly::DensePolynomial,
        //EqPolynomial dectecs when two vectors are equal
        //IdentityPolynomial converts binary vectors to field elements
        eq_poly::EqPolynomial, identity_poly::IdentityPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeStuff<T: CanonicalSerialize + CanonicalDeserialize> {
    /// C-sized vector of `dim_i` polynomials/commitments/openings
    pub(crate) dim: Vec<T>,
    /// C-sized vector of `read_cts_i` polynomials/commitments/openings
    pub(crate) read_cts: Vec<T>,
    /// C-sized vector of `E_i` polynomials/commitments/openings
    pub(crate) E_polys: Vec<T>,
    /// `num_memories`-sized vector of `final_cts_i` polynomials/commitments/openings
    pub(crate) final_cts: Vec<T>,

    a_init_final: VerifierComputedOpening<T>,
    v_init_final: VerifierComputedOpening<Vec<T>>,
}
//SurgePolynomials, SurgeOpenings, and SurgeCommitments are all types that are used to store polynomial data
//SurgePolynomials is a type that is used to store polynomial data
//SurgeOpenings is a type that is used to store openings
//SurgeCommitments is a type that is used to store commitments
pub type SurgePolynomials<F: JoltField> = SurgeStuff<MultilinearPolynomial<F>>;
pub type SurgeOpenings<F: JoltField> = SurgeStuff<F>;
pub type SurgeCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
    SurgeStuff<PCS::Commitment>;
//Initializable traits provides a way to create a SurgeStuff instance with the correct sizes based on the preprocessing data.
impl<const C: usize, const M: usize, F, T, Instruction>
    Initializable<T, SurgePreprocessing<F, Instruction, C, M>> for SurgeStuff<T>
where
    F: JoltField,
    T: CanonicalSerialize + CanonicalDeserialize + Default,
    Instruction: JoltInstruction + Default,
{   
    fn nitialize(_preprocessing: &SurgePreprocessing<F, Instruction, C, M>) -> Self {
        //num_memories is the number of memories in the instruction
        //C: Number of dimensions
        //M: Size of each subtable
        //F: Field type
        //T: Type of the polynomial data (polynomials, openings, commitments?)
        //Instruction: RISC-V instruction type
        let num_memories = C * Instruction::default().subtables::<F>(C, M).len();
        Self {
            //dim: dimension polynomials (dim_i)
            dim: std::iter::repeat_with(|| T::default()).take(C).collect(),
            //read_cts: read count polynomials (read_cts_i)
            read_cts: std::iter::repeat_with(|| T::default()).take(C).collect(),
            //final_cts: final count polynomials (final_cts_i)
            final_cts: std::iter::repeat_with(|| T::default()).take(C).collect(),
            //E_i polynomials (E_i)
            E_polys: std::iter::repeat_with(|| T::default())
                .take(num_memories)
                .collect(),
            a_init_final: None,
            v_init_final: None,
        }
    }
}

impl<T: CanonicalSerialize + CanonicalDeserialize> StructuredPolynomialData<T> for SurgeStuff<T> {
    fn read_write_values(&self) -> Vec<&T> {
        self.dim
            .iter()
            .chain(self.read_cts.iter())
            .chain(self.E_polys.iter())
            .collect()
    }

    fn init_final_values(&self) -> Vec<&T> {
        self.final_cts.iter().collect()
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        self.dim
            .iter_mut()
            .chain(self.read_cts.iter_mut())
            .chain(self.E_polys.iter_mut())
            .collect()
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        self.final_cts.iter_mut().collect()
    }
}

impl<F, PCS, Instruction, const C: usize, const M: usize, ProofTranscript: Transcript>
    MemoryCheckingProver<F, PCS, ProofTranscript>
    for SurgeProof<F, PCS, Instruction, C, M, ProofTranscript>
where
    F: JoltField,
    Instruction: JoltInstruction + Default + Sync,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
{   //The following codes define the type for the MemoryCheckingProver trait. 
    type ReadWriteGrandProduct = BatchedDenseGrandProduct<F>;
    type InitFinalGrandProduct = BatchedDenseGrandProduct<F>;

    type Polynomials = SurgePolynomials<F>;
    type Openings = SurgeOpenings<F>;
    type Commitments = SurgeCommitments<PCS, ProofTranscript>;
    type ExogenousOpenings = NoExogenousOpenings;

    type Preprocessing = SurgePreprocessing<F, Instruction, C, M>;

    /// The data associated with each memory slot. A triple (a, v, t) by default.
    type MemoryTuple = (F, F, F); // (a, v, t)

    //fingerprint is a function that computes the fingerprint of a memory tuple
    //inputs: a tuple of (a, v, t)
    //gamma: a field element
    //tau: a field element
    //returns: a field element (the fingerprint of the memory tuple)    
    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - *tau
    }
    //compute_leaves is a function that computes the fingerprints (hashes) for the memory operations
    //These fingerprints are used to construct grand product circuits for memory consistency checks.    
    #[tracing::instrument(skip_all, name = "Surge::compute_leaves")]
    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        _: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> ((Vec<F>, usize), (Vec<F>, usize)) {
        let gamma_squared = gamma.square();

        let num_lookups = polynomials.dim[0].len();

        let read_write_leaves: Vec<_> = (0..Self::num_memories())
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                //The following steps extract the polynomials for the current memory index
                //dim_index: the index of the dimension for the current memory index
                //read_cts: the read count polynomial for the current memory index
                //E_poly: the E polynomial for the current memory index
                //dim: the dimension polynomial for the current memory index
                //read_fingerprints: the fingerprints for the read operations
                //write_fingerprints: the fingerprints for the write operations
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let read_cts: &CompactPolynomial<u32, F> =
                    (&polynomials.read_cts[dim_index]).try_into().unwrap();
                let E_poly: &CompactPolynomial<u32, F> =
                    (&polynomials.E_polys[memory_index]).try_into().unwrap();
                let dim: &CompactPolynomial<u16, F> =
                    (&polynomials.dim[dim_index]).try_into().unwrap();
                //Compute the fingerprints for the read operations
                let read_fingerprints: Vec<F> = (0..num_lookups)
                    .map(|i| {
                        let a = dim[i];
                        let v = E_poly[i];
                        let t = read_cts[i];
                        t.field_mul(gamma_squared) + v.field_mul(*gamma) + F::from_u16(a) - *tau
                    })
                    .collect();
                //Compute the fingerprints for the write operations
                //Write fingerprints are computed by adding a constant to the read fingerprints
                //t_adjustment is the constant, which is gamma^2. 
                let t_adjustment = 1u64.field_mul(gamma_squared);
                let write_fingerprints = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| *read_fingerprint + t_adjustment)
                    .collect();

                vec![read_fingerprints, write_fingerprints]
            })
            .collect();
        
        // Compute fingerprints for initial and final memory states.    
        let init_final_leaves: Vec<_> = (0..Self::num_memories())
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let subtable_index = Self::memory_to_subtable_index(memory_index);
                // TODO(moodlezoup): Only need one init polynomial per subtable
                let init_fingerprints: Vec<F> = (0..M)
                    .map(|i| {
                        // 0 * gamma^2 +
                        preprocessing.materialized_subtables[subtable_index][i].field_mul(*gamma)
                            + F::from_u64(i as u64)
                            - *tau
                    })
                    .collect();
                let final_fingerprints = init_fingerprints
                    .iter()
                    .enumerate()
                    .map(|(i, init_fingerprint)| {
                        let final_cts: &CompactPolynomial<u32, F> =
                            (&polynomials.final_cts[dim_index]).try_into().unwrap();
                        *init_fingerprint + final_cts[i].field_mul(gamma_squared)
                    })
                    .collect();

                vec![init_fingerprints, final_fingerprints]
            })
            .collect();

        // TODO(moodlezoup): avoid concat
        (
            (read_write_leaves.concat(), 2 * Self::num_memories()),
            (init_final_leaves.concat(), 2 * Self::num_memories()),
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"SurgeMemCheck"
    }
}

impl<F, PCS, Instruction, const C: usize, const M: usize, ProofTranscript>
    MemoryCheckingVerifier<F, PCS, ProofTranscript>
    for SurgeProof<F, PCS, Instruction, C, M, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    Instruction: JoltInstruction + Default + Sync,
    ProofTranscript: Transcript,
{
    //compute_verifier_openings is a function that the verifier can calculate itself without the prover's help. 
    fn compute_verifier_openings(
        openings: &mut Self::Openings,
        _preprocessing: &Self::Preprocessing,
        _r_read_write: &[F],
        r_init_final: &[F],
    ) {
        //The following steps compute the openings for the initial and final memory states. 
        //r_init_final: the openings for the initial memory states
        //openings.a_init_final: the openings for the initial memory states, 
        // which is the identity polynomial of the openings for the initial memory states
        //openings.v_init_final: the openings for the initial memory states, 
        // which is the evaluation of the subtable polynomials at the openings for the initial memory states
        openings.a_init_final =
            Some(IdentityPolynomial::new(r_init_final.len()).evaluate(r_init_final));
        openings.v_init_final = Some(
            Instruction::default()
                .subtables(C, M)
                .iter()
                .map(|(subtable, _)| subtable.evaluate_mle(r_init_final))
                .collect(),
        );
    }

    //Reconstruct the memory tuples for the read operations from the polynomial openings
    fn read_tuples(
        _preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..Self::num_memories())//Iterate over all the memory slots
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    openings.dim[dim_index], //address
                    openings.E_polys[memory_index], //value
                    openings.read_cts[dim_index], //timestamp for read
                )
            })
            .collect()
    }
    //Reconstruct the memory tuples for the write operations from the polynomial openings
    fn write_tuples(
        _preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..Self::num_memories())//Iterate over all the memory slots
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    openings.dim[dim_index],//address
                    openings.E_polys[memory_index],//value
                    openings.read_cts[dim_index] + F::one(),//timestamp for write, which is the timestamp for read plus 1
                )
            })
            .collect()
    }
    //Reconstruct the memory tuples for the initial memory states from the polynomial openings
    fn init_tuples(
        _preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::num_memories())//Iterate over all the memory slots
            .map(|memory_index| {
                (
                    a_init, //Address for all initial states
                    v_init[Self::memory_to_subtable_index(memory_index)], 
                    F::zero(),//timestamp 0 for all initial states
                )
            })
            .collect()
    }
    //Reconstruct the memory tuples for the final memory states from the polynomial openings
    fn final_tuples(
        _preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::num_memories())//Iterate over all the memory slots
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    a_init,//Address for all final states
                    v_init[Self::memory_to_subtable_index(memory_index)],//Value for all final states
                    openings.final_cts[dim_index],//Timestamp comes from the final count polynomial
                )
            })
            .collect()
    }
}

// Define the structure for the primary sumcheck proof
pub struct SurgePrimarySumcheck<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,//The actual sumcheck proof
    num_rounds: usize,//The number of rounds in the primary sumcheck
    claimed_evaluation: F,//The claimed evaluation of the primary sumcheck
    E_poly_openings: Vec<F>,//The openings for the E polynomials
    _marker: PhantomData<ProofTranscript>,//A marker for the proof transcript
}
//This struct holds precomputed data that doesn't change during proof generation.       
pub struct SurgePreprocessing<F, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    Instruction: JoltInstruction + Default,
{
    _instruction: PhantomData<Instruction>,//A marker for the instruction
    _field: PhantomData<F>,//A marker for the field
    materialized_subtables: Vec<Vec<u32>>,//Precomputed lookup tables for RISC-V instructions
}

//This struct holds the proof data that changes during proof generation. 
#[allow(clippy::type_complexity)]
pub struct SurgeProof<F, PCS, Instruction, const C: usize, const M: usize, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    Instruction: JoltInstruction + Default,
    ProofTranscript: Transcript,
{
    _instruction: PhantomData<Instruction>,
    /// Commitments to all polynomials
    commitments: SurgeCommitments<PCS, ProofTranscript>,

    /// Primary collation sumcheck proof
    primary_sumcheck: SurgePrimarySumcheck<F, ProofTranscript>,

    memory_checking:
        MemoryCheckingProof<F, PCS, SurgeOpenings<F>, NoExogenousOpenings, ProofTranscript>,
}

impl<F, Instruction, const C: usize, const M: usize> SurgePreprocessing<F, Instruction, C, M>
where
    F: JoltField,
    Instruction: JoltInstruction + Default + Sync,
{
    #[tracing::instrument(skip_all, name = "Surge::preprocess")]
    pub fn preprocess() -> Self {
        //The core preprocessing step is to create the lookup tables for the RISC-V instructions
        let instruction = Instruction::default(); //create a default intance of the instruction type

        let materialized_subtables = instruction
            .subtables::<F>(C, M) //Get the subtable needed for the instruction
            .par_iter() //Parallelize the iteration
            .map(|(subtable, _)| subtable.materialize(M)) //Materialize the subtable
            .collect(); //Collect the materialized subtable into a vector

        // TODO(moodlezoup): do PCS setup here

        Self {
            _instruction: PhantomData,
            _field: PhantomData,
            materialized_subtables,
        }
    }
}

impl<F, PCS, Instruction, const C: usize, const M: usize, ProofTranscript>
    SurgeProof<F, PCS, Instruction, C, M, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    Instruction: JoltInstruction + Default + Sync,
    ProofTranscript: Transcript,
{
    // TODO(moodlezoup): We can be more efficient (use fewer memories) if we use subtable_indices
    fn num_memories() -> usize {
        C * Instruction::default().subtables::<F>(C, M).len()
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, NUM_SUBTABLES)
    fn memory_to_subtable_index(i: usize) -> usize {
        i / C
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, C)
    fn memory_to_dimension_index(i: usize) -> usize {
        i % C
    }

    fn protocol_name() -> &'static [u8] {
        b"Surge"
    }

    /// Computes the maximum number of group generators needed to commit to Surge polynomials
    /// using Hyrax, given `M` and the maximum number of lookups.
    pub fn num_generators(max_num_lookups: usize) -> usize {
        std::cmp::max(
            max_num_lookups.next_power_of_two(),
            (M * Self::num_memories()).next_power_of_two(),
        )
    }

    #[tracing::instrument(skip_all, name = "Surge::prove")]
    pub fn prove(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        generators: &PCS::Setup,
        ops: Vec<Instruction>,
    ) -> (Self, Option<ProverDebugInfo<F, ProofTranscript>>) {
        let mut transcript = ProofTranscript::new(b"Surge transcript");
        let mut opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
            ProverOpeningAccumulator::new();
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let num_lookups = ops.len().next_power_of_two();
        let polynomials = Self::generate_witness(preprocessing, &ops);

        let mut commitments = SurgeCommitments::<PCS, ProofTranscript>::initialize(preprocessing);
        let trace_polys = polynomials.read_write_values();
        let trace_comitments = PCS::batch_commit(&trace_polys, generators);
        commitments
            .read_write_values_mut()
            .into_iter()
            .zip(trace_comitments.into_iter())
            .for_each(|(dest, src)| *dest = src);
        commitments.final_cts = PCS::batch_commit(&polynomials.final_cts, generators);

        let num_rounds = num_lookups.log_2();
        let instruction = Instruction::default();

        // TODO(sragss): Commit some of this stuff to transcript?

        // Primary sumcheck
        let r_primary_sumcheck: Vec<F> = transcript.challenge_vector(num_rounds);
        let eq = MultilinearPolynomial::from(EqPolynomial::evals(&r_primary_sumcheck));
        let sumcheck_claim: F = Self::compute_primary_sumcheck_claim(&polynomials, &eq);

        transcript.append_scalar(&sumcheck_claim);
        let mut combined_sumcheck_polys = polynomials.E_polys.clone();
        combined_sumcheck_polys.push(eq);

        let combine_lookups_eq = |vals: &[F]| -> F {
            let vals_no_eq: &[F] = &vals[0..(vals.len() - 1)];
            let eq = vals[vals.len() - 1];
            instruction.combine_lookups(vals_no_eq, C, M) * eq
        };

        let (primary_sumcheck_proof, r_z, mut sumcheck_openings) =
            SumcheckInstanceProof::<F, ProofTranscript>::prove_arbitrary::<_>(
                &sumcheck_claim,
                num_rounds,
                &mut combined_sumcheck_polys,
                combine_lookups_eq,
                instruction.g_poly_degree(C) + 1, // combined degree + eq term
                &mut transcript,
            );

        // Remove EQ
        let _ = combined_sumcheck_polys.pop();
        let _ = sumcheck_openings.pop();
        opening_accumulator.append(
            &polynomials.E_polys.iter().collect::<Vec<_>>(),
            DensePolynomial::new(EqPolynomial::evals(&r_z)),
            r_z.clone(),
            &sumcheck_openings,
            &mut transcript,
        );

        let primary_sumcheck = SurgePrimarySumcheck {
            claimed_evaluation: sumcheck_claim,
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            E_poly_openings: sumcheck_openings,
            _marker: PhantomData,
        };

        let memory_checking = SurgeProof::prove_memory_checking(
            generators,
            preprocessing,
            &polynomials,
            &JoltPolynomials::default(), // Hack: required by the memory-checking trait, but unused in Surge
            &mut opening_accumulator,
            &mut transcript,
        );

        let proof = SurgeProof {
            _instruction: PhantomData,
            commitments,
            primary_sumcheck,
            memory_checking,
        };
        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript,
            opening_accumulator,
        });
        #[cfg(not(test))]
        let debug_info = None;

        (proof, debug_info)
    }

    pub fn verify(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        generators: &PCS::Setup,
        proof: SurgeProof<F, PCS, Instruction, C, M, ProofTranscript>,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript>>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Surge transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();
        #[cfg(test)]
        if let Some(debug_info) = _debug_info {
            transcript.compare_to(debug_info.transcript);
            opening_accumulator.compare_to(debug_info.opening_accumulator, generators);
        }

        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);
        let instruction = Instruction::default();

        let r_primary_sumcheck = transcript.challenge_vector(proof.primary_sumcheck.num_rounds);

        transcript.append_scalar(&proof.primary_sumcheck.claimed_evaluation);
        let primary_sumcheck_poly_degree = instruction.g_poly_degree(C) + 1;
        let (claim_last, r_z) = proof.primary_sumcheck.sumcheck_proof.verify(
            proof.primary_sumcheck.claimed_evaluation,
            proof.primary_sumcheck.num_rounds,
            primary_sumcheck_poly_degree,
            &mut transcript,
        )?;

        let eq_eval = EqPolynomial::new(r_primary_sumcheck.to_vec()).evaluate(&r_z);
        assert_eq!(
            eq_eval * instruction.combine_lookups(&proof.primary_sumcheck.E_poly_openings, C, M),
            claim_last,
            "Primary sumcheck check failed."
        );

        opening_accumulator.append(
            &proof.commitments.E_polys.iter().collect::<Vec<_>>(),
            r_z.clone(),
            &proof
                .primary_sumcheck
                .E_poly_openings
                .iter()
                .collect::<Vec<_>>(),
            &mut transcript,
        );

        Self::verify_memory_checking(
            preprocessing,
            generators,
            proof.memory_checking,
            &proof.commitments,
            &JoltCommitments::<PCS, ProofTranscript>::default(),
            &mut opening_accumulator,
            &mut transcript,
        )
    }

    #[tracing::instrument(skip_all, name = "Surge::construct_polys")]
    fn generate_witness(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        ops: &[Instruction],
    ) -> SurgePolynomials<F> {
        let num_lookups = ops.len().next_power_of_two();
        let mut dim: Vec<Vec<u16>> = vec![vec![0; num_lookups]; C];

        let mut read_cts = vec![vec![0u32; num_lookups]; C];
        let mut final_cts = vec![vec![0u32; M]; C];
        let log_M = ark_std::log2(M) as usize;

        for (op_index, op) in ops.iter().enumerate() {
            let access_sequence = op.to_indices(C, log_M);
            assert_eq!(access_sequence.len(), C);

            for dimension_index in 0..C {
                let memory_address = access_sequence[dimension_index];
                debug_assert!(memory_address < M);

                dim[dimension_index][op_index] = memory_address as u16;

                let ts = final_cts[dimension_index][memory_address];
                read_cts[dimension_index][op_index] = ts;
                let write_timestamp = ts + 1;
                final_cts[dimension_index][memory_address] = write_timestamp;
            }
        }

        // num_ops is padded to the nearest power of 2 for the usage of DensePolynomial. We cannot just fill
        // in zeros for read_cts and final_cts as this implicitly specifies a read at address 0. The prover
        // and verifier plumbing assume write_ts(r) = read_ts(r) + 1. This will not hold unless we update
        // the final_cts for these phantom reads.
        for fake_ops_index in ops.len()..num_lookups {
            for dimension_index in 0..C {
                let memory_address = 0;
                let ts = final_cts[dimension_index][memory_address];
                read_cts[dimension_index][fake_ops_index] = ts;
                let write_timestamp = ts + 1;
                final_cts[dimension_index][memory_address] = write_timestamp;
            }
        }

        // Construct E
        let mut E_i_evals = Vec::with_capacity(Self::num_memories());
        for E_index in 0..Self::num_memories() {
            let mut E_evals = Vec::with_capacity(num_lookups);
            for op_index in 0..num_lookups {
                let dimension_index = Self::memory_to_dimension_index(E_index);
                let subtable_index = Self::memory_to_subtable_index(E_index);

                let eval_index = dim[dimension_index][op_index];
                let eval =
                    preprocessing.materialized_subtables[subtable_index][eval_index as usize];
                E_evals.push(eval);
            }
            E_i_evals.push(E_evals);
        }

        let E_polys: Vec<MultilinearPolynomial<F>> = E_i_evals
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();
        let dim: Vec<MultilinearPolynomial<F>> =
            dim.into_iter().map(MultilinearPolynomial::from).collect();
        let read_cts: Vec<MultilinearPolynomial<F>> = read_cts
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();
        let final_cts: Vec<MultilinearPolynomial<F>> = final_cts
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        SurgePolynomials {
            dim,
            read_cts,
            final_cts,
            E_polys,
            a_init_final: None,
            v_init_final: None,
        }
    }

    #[tracing::instrument(skip_all, name = "Surge::compute_primary_sumcheck_claim")]
    fn compute_primary_sumcheck_claim(
        polys: &SurgePolynomials<F>,
        eq: &MultilinearPolynomial<F>,
    ) -> F {
        let g_operands = &polys.E_polys;
        let hypercube_size = g_operands[0].len();
        g_operands
            .iter()
            .for_each(|operand| assert_eq!(operand.len(), hypercube_size));

        let instruction = Instruction::default();

        (0..hypercube_size)
            .into_par_iter()
            .map(|eval_index| {
                let g_operands: Vec<F> = (0..Self::num_memories())
                    .map(|memory_index| g_operands[memory_index].get_coeff(eval_index))
                    .collect();
                eq.get_coeff(eval_index) * instruction.combine_lookups(&g_operands, C, M)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::SurgePreprocessing;
    use crate::utils::transcript::KeccakTranscript;
    use crate::{
        jolt::instruction::xor::XORInstruction,
        lasso::surge::SurgeProof,
        poly::commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
    };
    use ark_bn254::{Bn254, Fr};
    use ark_std::test_rng;
    use rand_core::RngCore;

    #[test]
    fn surge_32_e2e() {
        let mut rng = test_rng();
        const WORD_SIZE: usize = 32;
        const C: usize = 4;
        const M: usize = 1 << 16;
        const NUM_OPS: usize = 1024;

        let ops = std::iter::repeat_with(|| {
            XORInstruction::<WORD_SIZE>(rng.next_u32() as u64, rng.next_u32() as u64)
        })
        .take(NUM_OPS)
        .collect();

        let preprocessing = SurgePreprocessing::preprocess();
        let generators = HyperKZG::<_, KeccakTranscript>::setup(M);
        let (proof, debug_info) = SurgeProof::<
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            XORInstruction<WORD_SIZE>,
            C,
            M,
            KeccakTranscript,
        >::prove(&preprocessing, &generators, ops);

        SurgeProof::verify(&preprocessing, &generators, proof, debug_info).expect("should work");
    }

    #[test]
    fn surge_32_e2e_non_pow_2() {
        let mut rng = test_rng();
        const WORD_SIZE: usize = 32;
        const C: usize = 4;
        const M: usize = 1 << 16;

        const NUM_OPS: usize = 1000;

        let ops = std::iter::repeat_with(|| {
            XORInstruction::<WORD_SIZE>(rng.next_u32() as u64, rng.next_u32() as u64)
        })
        .take(NUM_OPS)
        .collect();

        let preprocessing = SurgePreprocessing::preprocess();
        let generators = HyperKZG::<_, KeccakTranscript>::setup(M);
        let (proof, debug_info) = SurgeProof::<
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            XORInstruction<WORD_SIZE>,
            C,
            M,
            KeccakTranscript,
        >::prove(&preprocessing, &generators, ops);

        SurgeProof::verify(&preprocessing, &generators, proof, debug_info).expect("should work");
    }
}
