"""Level 2 Module: Control Flow Analysis.

Provides deterministic analysis of control flow:
- Basic block detection
- CFG construction
- Function detection
- Loop detection
"""

from typing import Optional
from collections import defaultdict

from levels.level1_assembly.types import Level1Output, ControlFlowType
from levels.level2_ir.types import (
    Level2Input,
    Level2Output,
    BasicBlock,
    CFGEdge,
    Function,
    CallEdge,
    Loop,
    EntryType,
    ExitType,
    EdgeType,
    LoopType,
)


class Level2Module:
    """Deterministic control flow analyzer.
    
    Analyzes sequences of Level 1 outputs to produce:
    - Basic blocks
    - Control flow graph
    - Functions
    - Loops
    """
    
    def __init__(self):
        self._block_id_counter = 0
    
    def analyze(self, input: Level2Input) -> Level2Output:
        """Analyze control flow from instruction sequence."""
        if not input.instructions:
            return Level2Output(is_uncertain=True, uncertainty_reason="empty input")
        
        # Reset state
        self._block_id_counter = 0
        
        # Step 1: Detect basic block boundaries
        block_starts = self._find_block_starts(input.instructions, input.entry_point)
        
        # Step 2: Create basic blocks
        basic_blocks = self._create_basic_blocks(input.instructions, block_starts)
        
        # Step 3: Build CFG edges
        cfg_edges = self._build_cfg_edges(basic_blocks)
        
        # Step 4: Detect functions
        functions, call_edges = self._detect_functions(basic_blocks, cfg_edges)
        
        # Step 5: Detect loops
        loops = self._detect_loops(basic_blocks, cfg_edges)
        
        return Level2Output(
            basic_blocks=basic_blocks,
            cfg_edges=cfg_edges,
            functions=functions,
            call_edges=call_edges,
            loops=loops,
        )
    
    def _find_block_starts(
        self, 
        instructions: list[Level1Output],
        entry_point: int
    ) -> dict[int, EntryType]:
        """Find all basic block start offsets."""
        block_starts: dict[int, EntryType] = {}
        
        # Entry point is always a block start
        block_starts[entry_point] = EntryType.FUNCTION_START
        
        # Build offset -> instruction map
        offset_to_instr: dict[int, Level1Output] = {}
        offsets = []
        for instr in instructions:
            offset = instr.instruction.offset
            offset_to_instr[offset] = instr
            offsets.append(offset)
        
        offsets.sort()
        
        for i, offset in enumerate(offsets):
            instr = offset_to_instr[offset]
            cf = instr.control_flow
            
            if cf is None:
                continue
            
            # Get next instruction offset
            next_offset = offsets[i + 1] if i + 1 < len(offsets) else None
            
            if cf.type == ControlFlowType.JUMP:
                # Unconditional jump - target is block start
                target = self._parse_target(cf.target_expr)
                if target is not None and target not in block_starts:
                    block_starts[target] = EntryType.JUMP_TARGET
                # Instruction after jump is block start (if reachable)
                if next_offset is not None and next_offset not in block_starts:
                    block_starts[next_offset] = EntryType.FALL_THROUGH
            
            elif cf.type == ControlFlowType.CONDITIONAL:
                # Conditional jump - both target and fall-through are block starts
                target = self._parse_target(cf.target_expr)
                if target is not None and target not in block_starts:
                    block_starts[target] = EntryType.JUMP_TARGET
                if next_offset is not None and next_offset not in block_starts:
                    block_starts[next_offset] = EntryType.FALL_THROUGH
            
            elif cf.type == ControlFlowType.CALL:
                # Call - target is function start
                target = self._parse_target(cf.target_expr)
                if target is not None and target not in block_starts:
                    block_starts[target] = EntryType.FUNCTION_START
                # Instruction after call is block start
                if next_offset is not None and next_offset not in block_starts:
                    block_starts[next_offset] = EntryType.FALL_THROUGH
            
            elif cf.type == ControlFlowType.RETURN:
                # Return - instruction after is block start (may be unreachable)
                if next_offset is not None and next_offset not in block_starts:
                    block_starts[next_offset] = EntryType.UNKNOWN
        
        return block_starts
    
    def _parse_target(self, target_expr: Optional[str]) -> Optional[int]:
        """Parse target expression to get offset."""
        if target_expr is None:
            return None
        
        # Handle hex format
        if target_expr.startswith("0x"):
            try:
                return int(target_expr, 16)
            except ValueError:
                return None
        
        # Handle decimal
        try:
            return int(target_expr)
        except ValueError:
            return None
    
    def _create_basic_blocks(
        self,
        instructions: list[Level1Output],
        block_starts: dict[int, EntryType]
    ) -> list[BasicBlock]:
        """Create basic blocks from instructions."""
        if not instructions:
            return []
        
        # Sort block starts
        sorted_starts = sorted(block_starts.keys())
        
        # Build offset -> instruction map
        offset_to_instr: dict[int, Level1Output] = {
            instr.instruction.offset: instr for instr in instructions
        }
        offsets = sorted(offset_to_instr.keys())
        
        blocks = []
        
        for i, start in enumerate(sorted_starts):
            # Find end offset (next block start or end of instructions)
            if i + 1 < len(sorted_starts):
                end_offset = sorted_starts[i + 1] - 1
            else:
                end_offset = offsets[-1] if offsets else start
            
            # Collect instructions in this block
            block_instrs = []
            for offset in offsets:
                if start <= offset <= end_offset:
                    if offset in offset_to_instr:
                        block_instrs.append(offset_to_instr[offset])
            
            if not block_instrs:
                continue
            
            # Determine exit type from last instruction
            exit_type = self._determine_exit_type(block_instrs[-1])
            
            block = BasicBlock(
                id=self._next_block_id(),
                start_offset=start,
                end_offset=block_instrs[-1].instruction.offset,
                instructions=block_instrs,
                entry_type=block_starts[start],
                exit_type=exit_type,
            )
            blocks.append(block)
        
        return blocks
    
    def _determine_exit_type(self, last_instr: Level1Output) -> ExitType:
        """Determine how a block exits based on last instruction."""
        cf = last_instr.control_flow
        
        if cf is None:
            return ExitType.FALL_THROUGH
        
        if cf.type == ControlFlowType.SEQUENTIAL:
            return ExitType.FALL_THROUGH
        elif cf.type == ControlFlowType.JUMP:
            return ExitType.UNCONDITIONAL_JUMP
        elif cf.type == ControlFlowType.CONDITIONAL:
            return ExitType.CONDITIONAL_JUMP
        elif cf.type == ControlFlowType.CALL:
            return ExitType.CALL
        elif cf.type == ControlFlowType.RETURN:
            return ExitType.RETURN
        else:
            return ExitType.UNKNOWN
    
    def _build_cfg_edges(self, blocks: list[BasicBlock]) -> list[CFGEdge]:
        """Build CFG edges between basic blocks."""
        edges = []
        
        # Build offset -> block map
        offset_to_block: dict[int, BasicBlock] = {}
        for block in blocks:
            offset_to_block[block.start_offset] = block
        
        # Build sorted list of block starts for finding next block
        sorted_starts = sorted(offset_to_block.keys())
        
        for block in blocks:
            last_instr = block.instructions[-1] if block.instructions else None
            if last_instr is None:
                continue
            
            cf = last_instr.control_flow
            
            # Find next block (fall-through)
            next_block = self._find_next_block(block.end_offset, sorted_starts, offset_to_block)
            
            if cf is None or cf.type == ControlFlowType.SEQUENTIAL:
                # Fall through to next block
                if next_block:
                    edges.append(CFGEdge(
                        source_block=block.id,
                        target_block=next_block.id,
                        edge_type=EdgeType.FALL_THROUGH,
                    ))
            
            elif cf.type == ControlFlowType.JUMP:
                # Unconditional jump
                target = self._parse_target(cf.target_expr)
                if target is not None and target in offset_to_block:
                    edges.append(CFGEdge(
                        source_block=block.id,
                        target_block=offset_to_block[target].id,
                        edge_type=EdgeType.UNCONDITIONAL,
                    ))
            
            elif cf.type == ControlFlowType.CONDITIONAL:
                # Conditional jump - two edges
                target = self._parse_target(cf.target_expr)
                if target is not None and target in offset_to_block:
                    edges.append(CFGEdge(
                        source_block=block.id,
                        target_block=offset_to_block[target].id,
                        edge_type=EdgeType.CONDITIONAL_TRUE,
                        condition=cf.condition,
                    ))
                if next_block:
                    edges.append(CFGEdge(
                        source_block=block.id,
                        target_block=next_block.id,
                        edge_type=EdgeType.CONDITIONAL_FALSE,
                    ))
            
            elif cf.type == ControlFlowType.CALL:
                # Call - edge to callee and fall-through
                target = self._parse_target(cf.target_expr)
                if target is not None and target in offset_to_block:
                    edges.append(CFGEdge(
                        source_block=block.id,
                        target_block=offset_to_block[target].id,
                        edge_type=EdgeType.CALL,
                    ))
                if next_block:
                    edges.append(CFGEdge(
                        source_block=block.id,
                        target_block=next_block.id,
                        edge_type=EdgeType.FALL_THROUGH,
                    ))
            
            # RETURN has no outgoing edges
        
        return edges
    
    def _find_next_block(
        self,
        current_end: int,
        sorted_starts: list[int],
        offset_to_block: dict[int, BasicBlock]
    ) -> Optional[BasicBlock]:
        """Find the next block after current_end."""
        for start in sorted_starts:
            if start > current_end:
                return offset_to_block.get(start)
        return None
    
    def _detect_functions(
        self,
        blocks: list[BasicBlock],
        edges: list[CFGEdge]
    ) -> tuple[list[Function], list[CallEdge]]:
        """Detect function boundaries."""
        functions = []
        call_edges = []
        
        # Find function entry blocks (FUNCTION_START entry type)
        entry_blocks = [b for b in blocks if b.entry_type == EntryType.FUNCTION_START]
        
        # Build adjacency list
        successors: dict[int, list[int]] = defaultdict(list)
        for edge in edges:
            if edge.edge_type != EdgeType.CALL:  # Don't follow calls for function boundaries
                successors[edge.source_block].append(edge.target_block)
        
        # For each entry, find all reachable blocks (within function)
        for entry in entry_blocks:
            # BFS to find all blocks in function
            visited = set()
            queue = [entry.id]
            exit_blocks = []
            calls = []
            
            while queue:
                block_id = queue.pop(0)
                if block_id in visited:
                    continue
                visited.add(block_id)
                
                block = self._get_block_by_id(blocks, block_id)
                if block is None:
                    continue
                
                # Check if this is an exit block
                if block.exit_type == ExitType.RETURN:
                    exit_blocks.append(block_id)
                
                # Check for calls
                if block.exit_type == ExitType.CALL and block.instructions:
                    last_instr = block.instructions[-1]
                    if last_instr.control_flow:
                        target = self._parse_target(last_instr.control_flow.target_expr)
                        if target is not None:
                            calls.append(target)
                            call_edges.append(CallEdge(
                                caller_offset=entry.start_offset,
                                callee_offset=target,
                                call_site=block.end_offset,
                                is_direct=True,
                            ))
                
                # Add successors to queue
                for succ in successors[block_id]:
                    if succ not in visited:
                        queue.append(succ)
            
            func = Function(
                entry_offset=entry.start_offset,
                entry_block=entry.id,
                exit_blocks=exit_blocks,
                blocks=list(visited),
                calls=calls,
            )
            functions.append(func)
        
        return functions, call_edges
    
    def _get_block_by_id(self, blocks: list[BasicBlock], block_id: int) -> Optional[BasicBlock]:
        """Get block by ID."""
        for block in blocks:
            if block.id == block_id:
                return block
        return None
    
    def _detect_loops(
        self,
        blocks: list[BasicBlock],
        edges: list[CFGEdge]
    ) -> list[Loop]:
        """Detect natural loops in CFG."""
        if not blocks:
            return []
        
        loops = []
        
        # Compute dominators
        dominators = self._compute_dominators(blocks, edges)
        
        # Find back edges (edge A->B where B dominates A)
        for edge in edges:
            source = edge.source_block
            target = edge.target_block
            
            # Check if target dominates source (back edge)
            if target in dominators.get(source, set()):
                # Found a back edge - this is a loop
                loop_body = self._find_loop_body(source, target, blocks, edges)
                
                loop = Loop(
                    header_block=target,
                    back_edge_block=source,
                    body_blocks=loop_body,
                    loop_type=self._classify_loop(target, blocks, edges),
                )
                loops.append(loop)
        
        return loops
    
    def _compute_dominators(
        self,
        blocks: list[BasicBlock],
        edges: list[CFGEdge]
    ) -> dict[int, set[int]]:
        """Compute dominator sets for all blocks."""
        if not blocks:
            return {}
        
        # Initialize dominators
        all_block_ids = {b.id for b in blocks}
        entry_id = blocks[0].id  # Assume first block is entry
        
        dom: dict[int, set[int]] = {}
        dom[entry_id] = {entry_id}
        
        for block in blocks:
            if block.id != entry_id:
                dom[block.id] = all_block_ids.copy()
        
        # Build predecessor map
        predecessors: dict[int, list[int]] = defaultdict(list)
        for edge in edges:
            predecessors[edge.target_block].append(edge.source_block)
        
        # Iterate until fixed point
        changed = True
        while changed:
            changed = False
            for block in blocks:
                if block.id == entry_id:
                    continue
                
                # dom[n] = {n} ∪ ∩{dom[p] for p in predecessors[n]}
                preds = predecessors[block.id]
                if preds:
                    new_dom = set.intersection(*[dom[p] for p in preds if p in dom])
                    new_dom.add(block.id)
                    
                    if new_dom != dom[block.id]:
                        dom[block.id] = new_dom
                        changed = True
        
        return dom
    
    def _find_loop_body(
        self,
        back_edge_source: int,
        header: int,
        blocks: list[BasicBlock],
        edges: list[CFGEdge]
    ) -> list[int]:
        """Find all blocks in loop body using reverse reachability."""
        loop_body = {header, back_edge_source}
        
        # Build predecessor map (excluding back edge)
        predecessors: dict[int, list[int]] = defaultdict(list)
        for edge in edges:
            # Don't include the back edge itself
            if not (edge.source_block == back_edge_source and edge.target_block == header):
                predecessors[edge.target_block].append(edge.source_block)
        
        # Reverse BFS from back_edge_source
        worklist = [back_edge_source]
        while worklist:
            node = worklist.pop()
            for pred in predecessors[node]:
                if pred not in loop_body:
                    loop_body.add(pred)
                    worklist.append(pred)
        
        return list(loop_body)
    
    def _classify_loop(
        self,
        header: int,
        blocks: list[BasicBlock],
        edges: list[CFGEdge]
    ) -> LoopType:
        """Classify loop type based on structure."""
        header_block = self._get_block_by_id(blocks, header)
        if header_block is None:
            return LoopType.UNKNOWN
        
        # Check exit type of header
        if header_block.exit_type == ExitType.CONDITIONAL_JUMP:
            return LoopType.PRE_TESTED  # while loop
        
        # TODO: more sophisticated analysis for do-while, for loops
        return LoopType.UNKNOWN
    
    def _next_block_id(self) -> int:
        """Get next block ID."""
        block_id = self._block_id_counter
        self._block_id_counter += 1
        return block_id
